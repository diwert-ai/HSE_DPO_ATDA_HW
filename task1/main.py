import warnings

import torchvision.transforms.functional
warnings.filterwarnings("ignore")

import os
import math
from datetime import datetime

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as tvF
from torch.utils.data import DataLoader, Dataset
import pytorch_lightning as pl
from torchmetrics.classification import BinaryJaccardIndex, BinaryAccuracy
import torchvision

import albumentations as albu

import timm

from PIL import ImageDraw, ImageFont

class Config:
    data_dir = './CamVid/'
    log_font = ImageFont.truetype("/usr/share/fonts/truetype/freefont/FreeMono.ttf", 28, encoding="unic")
    num_log_images = 4
    top_bottom_k = 4
    devices = 1
    nodes = 1
    num_workers = 10
    accelerator = 'gpu'
    
    hparams = {
        'experiment': 'Debug',
        'arch': 'UNet',
        'backbone': 'mixnet_l',
        'loss': 'Dice+BCE',
        'epochs' : 70,
        'train_batch_size' : 16,
        'valid_batch_size' : 8,
        'test_batch_size' : 8,
        'lr' : 1e-4,
        'weight_decay': 1e-4,
        'gamma':  0.99,
        'seed': 42,
    }



class UNET(nn.Module):

    def __init__(self, n_classes, last_act='', pretrained=True, backbone='resnet18'):
        super().__init__()

        # в init определяются подмодули и буферы
        self.encoder = timm.create_model(backbone, features_only=True, pretrained=pretrained)
        
        acts = self.encoder(torch.randn(1, 3, 480, 384))[:5]

        ch_in = acts[4].shape[1] + acts[3].shape[1]
        self.decoder_conv1 = nn.Conv2d(ch_in, 256, (3, 3), padding=1)

        ch_in = 256 + acts[2].shape[1]
        self.decoder_conv2 = nn.Conv2d(ch_in, 128, (3, 3), padding=1)

        ch_in = 128 + acts[1].shape[1]
        self.decoder_conv3 = nn.Conv2d(ch_in, 64, (3, 3), padding=1)

        ch_in = 64 + acts[0].shape[1]
        self.decoder_conv4 = nn.Conv2d(ch_in, 64, (3, 3), padding=1)

        # kernel_size = (1, 1) превращает Conv2d слой по сути в Linear слой,
        # который обрабатывает каждый пиксель в отдельности
        self.final = nn.Conv2d(64, n_classes, (1, 1), padding=0)

        self.last_act = None
        if last_act == 'sigmoid':
            # binary or multilabel classification
            self.last_act = nn.Sigmoid()
        elif last_act == 'softmax':
            # multiclass classification
            self.last_act = nn.Softmax()
        else:
            # logits
            assert last_act == ''

        cfg = self.encoder.pretrained_cfg if pretrained else self.encoder.default_cfg
        self.register_buffer('mean', torch.tensor(cfg['mean']).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor(cfg['std']).view(1, 3, 1, 1))

    def forward(self, x):
        """
        Args:
            x (Tensor): of shape (b, 3, H, W)
        Returns:
            Tensor of shape (b, n_classes, H, W)
            that containes either logits or probs (depends on final_act)
        """
        b, _3, H, W = x.shape

        x = (x - self.mean) / self.std

        h1, h2, h3, h4, h5 = self.encoder(x)

        out = torch.concatenate([h4, F.interpolate(h5, size=h4.shape[-2:], mode='bilinear')], dim=1)
        out = F.relu(self.decoder_conv1(out))
        out = torch.concatenate([h3, F.interpolate(out, size=h3.shape[-2:], mode='bilinear')], dim=1)
        out = F.relu(self.decoder_conv2(out))
        out = torch.concatenate([h2, F.interpolate(out, size=h2.shape[-2:], mode='bilinear')], dim=1)
        out = F.relu(self.decoder_conv3(out))
        out = torch.concatenate([h1, F.interpolate(out, size=h1.shape[-2:], mode='bilinear')], dim=1)
        out = F.relu(self.decoder_conv4(out))

        out = F.interpolate(out, size=(H, W), mode='bilinear') # (b, 64, H, W)
        out = self.final(out) # (b, n_classes, H, W)

        if self.last_act is not None:
            out = self.last_act(out)

        return out
    
def dice_score(output, target, eps=1e-7):
  """
  Args:
    output (Tensor): of shape (b, n_classes, h, w)
    target (Tensor): of shape (b, n_classes, h, w)
  """
  b, n_classes, h, w = output.shape
  union = torch.sum(output + target) * b * n_classes
  eps = eps if union < eps else 0.0
  dice_score = (2.0 * torch.sum(output * target) + eps) / (union + eps)
  return dice_score

def dice_loss(output, target):
  return 1 - dice_score(output, target)

class CamVidDataset(Dataset):
    """CamVid Dataset. Read images, apply augmentation and preprocessing transformations.

    Args:
        images_dir (str): path to images folder
        masks_dir (str): path to segmentation masks folder
        class_values (list): values of classes to extract from segmentation mask
        augmentation (albumentations.Compose): data transfromation pipeline
            (e.g. flip, scale, etc.)
        preprocessing (albumentations.Compose): data preprocessing
            (e.g. normalization, shape manipulation, etc.)

    """

    CLASSES = ['sky', 'building', 'pole', 'road', 'pavement',
               'tree', 'signsymbol', 'fence', 'car',
               'pedestrian', 'bicyclist', 'unlabelled']

    def __init__(
            self,
            images_dir,
            masks_dir,
            classes=None,
            augmentation=None,
            preprocessing=None,
    ):
        self.ids = os.listdir(images_dir)
        self.images_fps = [os.path.join(images_dir, image_id) for image_id in self.ids]
        self.masks_fps = [os.path.join(masks_dir, image_id) for image_id in self.ids]

        # convert str names to class values on masks
        self.class_values = [self.CLASSES.index(cls.lower()) for cls in classes]

        self.augmentation = augmentation
        self.preprocessing = preprocessing

    def __getitem__(self, i):

        # read data
        image = cv2.imread(self.images_fps[i])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(self.masks_fps[i], 0)

        # extract certain classes from mask (e.g. cars)
        masks = [(mask == v) for v in self.class_values]
        mask = np.stack(masks, axis=-1).astype('float')

        # apply augmentations
        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']

        # apply preprocessing
        if self.preprocessing:
            sample = self.preprocessing(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']

        return image, mask

    def __len__(self):
        return len(self.ids)
    
def get_training_augmentation():
    train_transform = [

        albu.HorizontalFlip(p=0.5),

        albu.ShiftScaleRotate(scale_limit=0.5, rotate_limit=0, shift_limit=0.1, p=1, border_mode=0),

        albu.PadIfNeeded(min_height=320, min_width=320, always_apply=True, border_mode=0),
        albu.RandomCrop(height=320, width=320, always_apply=True),

        albu.GaussNoise(p=0.2),
        albu.Perspective(p=0.5),

        albu.OneOf(
            [
                albu.CLAHE(p=1),
                albu.RandomBrightnessContrast(p=1, contrast_limit=0.0),
                albu.RandomGamma(p=1),
            ],
            p=0.9,
        ),

        albu.OneOf(
            [
                albu.Sharpen(p=1),
                albu.Blur(blur_limit=3, p=1),
                albu.MotionBlur(blur_limit=3, p=1),
            ],
            p=0.9,
        ),

        albu.OneOf(
            [
                albu.RandomBrightnessContrast(p=1, brightness_limit=0.0),
                albu.HueSaturationValue(p=1),
            ],
            p=0.9,
        ),
    ]
    return albu.Compose(train_transform)


def get_validation_augmentation():
    """Add paddings to make image shape divisible by 32"""
    test_transform = [
        albu.PadIfNeeded(384, 480)
    ]
    return albu.Compose(test_transform)


def to_tensor(x, **kwargs):
    return x.transpose(2, 0, 1).astype('float32')


def get_preprocessing(preprocessing_fn):
    """Construct preprocessing transform

    Args:
        preprocessing_fn (callbale): data normalization function
            (can be specific for each pretrained neural network)
    Return:
        transform: albumentations.Compose

    """

    _transform = [
        albu.Lambda(image=preprocessing_fn),
        albu.Lambda(image=to_tensor, mask=to_tensor),
    ]
    return albu.Compose(_transform)


class SegModel(pl.LightningModule):

    def __init__(self, data_dir):
        super().__init__()

        classes = ['car']

        def div_by_255(img, **kwargs):
            return img / 255.
        preprocessing_fn = div_by_255

        self.model = UNET(n_classes=1, last_act='sigmoid', backbone=Config.hparams['backbone'])

        self.bce_loss = nn.BCELoss()

        self.save_hyperparameters()

        self.train_dataset = CamVidDataset(
            os.path.join(data_dir, 'train'),
            os.path.join(data_dir, 'trainannot'),
            augmentation=get_training_augmentation(),
            preprocessing=get_preprocessing(preprocessing_fn),
            classes=classes,
        )

        self.val_dataset = CamVidDataset(
            os.path.join(data_dir, 'val'),
            os.path.join(data_dir, 'valannot'),
            augmentation=get_validation_augmentation(),
            preprocessing=get_preprocessing(preprocessing_fn),
            classes=classes,
        )

        self.test_dataset = CamVidDataset(
            os.path.join(data_dir, 'test'),
            os.path.join(data_dir, 'testannot'),
            augmentation=get_validation_augmentation(),
            preprocessing=get_preprocessing(preprocessing_fn),
            classes=classes,
        )

        self.iou = {
          'val': BinaryJaccardIndex(),
          'test': BinaryJaccardIndex(),
          'train': BinaryJaccardIndex(),
        }

        self.log_step_counter = {
            'val': 0,
            'test': 0,
            'train': 0,
        }

        self.log_k_step_counter = {
            'val': 0,
            'test': 0,
            'train': 0,
        }

        self.n_log = Config.num_log_images
        self.k = Config.top_bottom_k
        
        self.k_iou_batch = {
            'val': [list(), list()],
            'test': [list(), list()],
            'train': [list(), list()],
        }
        self.log_font = Config.log_font

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_nb):
        img, mask = batch
        out = self(img)
        loss = dice_loss(out, mask) + self.bce_loss(out, mask)
        self.iou['train'].to(self.device).update(out, mask.long())
        # self.log_batch_imgs(img, mask, out, 'train')
        self.eval_topk_bottomk(img, mask, out, 'train')
        return loss

    def on_train_epoch_end(self, **kwargs):
        self.on_eval_epoch_end('train', **kwargs)

    def log_topk_bottomk_images(self, stage):

        data_top = self.k_iou_batch[stage][0]
        data_bottom =  self.k_iou_batch[stage][1]

        log_images_top, log_images_bottom = [], []

        for i in range(len(data_top)):
            out_top, out_bottom = data_top[i][0]['pr'], data_bottom[i][0]['pr']
            mask_top, mask_bottom = data_top[i][0]['gt'], data_bottom[i][0]['gt']
            img_top, img_bottom = data_top[i][0]['img'], data_bottom[i][0]['img']
            acc_top, acc_bottom = 100 * data_top[i][0]['acc'], 100 * data_bottom[i][0]['acc']

            iou_top, iou_bottom = 100 * data_top[i][1], 100 * data_bottom[i][1]
                
            text_iou_top, text_iou_bottom = f'IoU {iou_top:.2f}%', f'IoU {iou_bottom:.2f}%'
            text_acc_top, text_acc_bottom = f'Acc {acc_top:.2f}%', f'Acc {acc_bottom:.2f}%'


            _, h, w = out_top.shape

            pred_mask_top, pred_mask_bottom = out_top.expand(3, h, w), out_bottom.expand(3, h, w)

            grid_list_top = [img_top, 
                             mask_top.expand(3, h, w),
                             pred_mask_top,
                             pred_mask_top.round()]
            
            grid_list_bottom = [img_bottom, 
                                mask_bottom.expand(3, h, w),
                                pred_mask_bottom,
                                pred_mask_bottom.round()]

            grid_top = torchvision.utils.make_grid(torch.stack(grid_list_top))
            grid_bottom = torchvision.utils.make_grid(torch.stack(grid_list_bottom))

            pil_grid_top = tvF.to_pil_image(grid_top)
            pil_grid_bottom = tvF.to_pil_image(grid_bottom)

            draw_top = ImageDraw.Draw(pil_grid_top)
            draw_bottom = ImageDraw.Draw(pil_grid_bottom)

            draw_top.text((10, 10), text_iou_top, fill=(255, 0, 0), font=self.log_font, stroke_width=1)
            draw_top.text((10, 40), text_acc_top, fill=(255, 0, 0), font=self.log_font, stroke_width=1)
                        
            draw_bottom.text((10, 10), text_iou_bottom, fill=(255, 0, 0), font=self.log_font, stroke_width=1)
            draw_bottom.text((10, 40), text_acc_bottom, fill=(255, 0, 0), font=self.log_font, stroke_width=1)

            grid_top = tvF.to_tensor(pil_grid_top)
            grid_bottom = tvF.to_tensor(pil_grid_bottom)

            log_images_top.append(grid_top)
            log_images_bottom.append(grid_bottom)
        
        log_image_top = torch.cat(log_images_top, dim=1)
        log_image_bottom = torch.cat(log_images_bottom, dim=1)

        step = self.log_k_step_counter[stage]

        self.logger.experiment.add_image(f'images_top_k_{self.k}/{stage}/{self.global_rank}', log_image_top, step)
        self.logger.experiment.add_image(f'images_bottom_k_{self.k}/{stage}/{self.global_rank}', log_image_bottom, step)
        
        self.log_k_step_counter[stage] = step + 1

    def log_batch_imgs(self, img, mask, out, stage):
        # print(self.local_rank)
        n = min(self.n_log, out.shape[0])
        log_images = []
        for i in range(n):
            _, h, w = out[i].shape
            pred_mask = out[i].expand(3, h, w)
            grid_list = [img[i], 
                         mask[i].expand(3, h, w),
                         pred_mask,
                         pred_mask.round()]
            grid = torchvision.utils.make_grid(torch.stack(grid_list))
            log_images.append(grid)
        log_image = torch.cat(log_images, dim=1)
        step = self.log_step_counter[stage]
        self.logger.experiment.add_image(f'images/{stage}/{self.global_rank}', log_image, step)
        self.log_step_counter[stage] = step + 1

    def eval_topk_bottomk(self, img, mask, out, stage):
        jacc = BinaryJaccardIndex().to(self.device)
        acc = BinaryAccuracy().to(self.device)
        def get_iou(i):
            iou = jacc(out[i], mask[i]).item()
            if math.isnan(iou):
                iou = 1.0
            return iou
        
        def get_record(i):
            return {'img': img[i],
                    'gt': mask[i],
                    'pr': out[i],
                    'acc': acc(out[i], mask[i]).item()}
        
        iou_batch = [(get_record(i), get_iou(i)) for i in range(mask.shape[0])]
        self.update_k_iou_batch(iou_batch, stage)

    def eval_step(self, batch, batch_idx, stage):
        img, mask = batch
        out = self(img)
        self.iou[stage].to(self.device).update(out, mask.long())
        # self.log_batch_imgs(img, mask, out, stage)
        self.eval_topk_bottomk(img, mask, out, stage)
    
    def update_k_iou_batch(self, iou_batch, stage):
        # is supposed to be not very large self.k

        iou_batch_sorted = sorted(iou_batch, key=lambda x: x[1], reverse=True)
        iou_batch_sorted_top = iou_batch_sorted[:self.k]
        iou_batch_sorted_bottom = iou_batch_sorted[:-self.k-1:-1]

        self.k_iou_batch[stage][0] = sorted(self.k_iou_batch[stage][0] + iou_batch_sorted_top,
                                            key=lambda x: x[1], 
                                            reverse=True)[:self.k]
        self.k_iou_batch[stage][1] = sorted(self.k_iou_batch[stage][1] + iou_batch_sorted_bottom,
                                            key=lambda x: x[1])[:self.k]

    def on_eval_epoch_end(self, stage, **kwargs):
        self.log(f"IOU/{stage}", self.iou[stage].to(self.device).compute(), prog_bar=True, logger=True, sync_dist=True)
        self.iou[stage].reset()
        self.log_topk_bottomk_images(stage)
        self.k_iou_batch[stage] = [list(), list()]
        
    def validation_step(self, batch, batch_idx):
        self.eval_step(batch, batch_idx, 'val')

    def test_step(self, batch, batch_idx):
        self.eval_step(batch, batch_idx, 'test')

    def on_validation_epoch_end(self, **kwargs):
        self.on_eval_epoch_end('val', **kwargs)

    def on_test_epoch_end(self, **kwargs):
        self.on_eval_epoch_end('test', **kwargs)

    def configure_optimizers(self):
        opt = torch.optim.Adam(self.model.parameters(),
                               lr=Config.hparams['lr'],
                               weight_decay=Config.hparams['weight_decay'],
                               )
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=10)
        return [opt], [sch]

    def train_dataloader(self):
        return DataLoader(self.train_dataset,
                          batch_size=Config.hparams['train_batch_size'], 
                          num_workers=Config.num_workers, 
                          shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset,
                          batch_size=Config.hparams['valid_batch_size'],
                          num_workers=Config.num_workers,
                          shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.test_dataset,
                          batch_size=Config.hparams['test_batch_size'],
                          num_workers=Config.num_workers,
                          shuffle=False)
    


if __name__ == '__main__':
    pl.seed_everything(42)
    seg_model = SegModel(Config.data_dir)
    checkpoint_callback = pl.callbacks.ModelCheckpoint(dirpath='checkpoints', 
                                                       monitor="IOU/val", 
                                                       mode="max"
                                                       )
    stamp = f'{datetime.now()}'.replace(':','-')
    logger = pl.loggers.TensorBoardLogger(save_dir="./",
                                          version=f'debug_{stamp}_{Config.hparams['backbone']}' 
                                          )
    
    trainer = pl.Trainer(callbacks=[checkpoint_callback],
                         accelerator = Config.accelerator,
                         devices=Config.devices,
                         num_nodes=Config.nodes,
                         max_epochs=Config.hparams['epochs'],
                         num_sanity_val_steps=0, 
                         # log_every_n_steps=6,
                         logger=logger,
                         # sync_batchnorm=True,
                         )
    

    
    trainer.fit(seg_model)
    trainer.test(ckpt_path='best')
