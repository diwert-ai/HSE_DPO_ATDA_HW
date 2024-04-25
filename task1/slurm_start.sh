#!/bin/bash -l
#SBATCH -J hse_appl
#SBATCH -o hse_appl-%j.out
#SBATCH -e hse_appl-%j.err
#SBATCH --nodes=2
#SBATCH --gres=gpu:2
#SBATCH --ntasks-per-node=2
#SBATCH --mem=0
#SBATCH -t 0-02:00:00

cd /home/ZotovA/work/HSE_DPO_ATDA_HW
source /opt/anaconda3/bin/activate hse_appl

# debugging flags (optional)
# export NCCL_DEBUG=INFO
# export PYTHONFAULTHANDLER=1

srun python main.py