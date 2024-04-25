#!/bin/bash
set -x
source /opt/anaconda3/bin/activate
conda create -n hse_appl python=3.12 --yes
conda activate hse_appl
pip install -r requirements.txt
conda env export > environment.yml
set +x