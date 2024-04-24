#!/bin/bash
set -x
source /opt/anaconda3/bin/activate hse_appl
tensorboard --logdir=lightning_logs --samples_per_plugin "images=100"
set +x