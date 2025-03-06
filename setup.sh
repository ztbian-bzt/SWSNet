#!/usr/bin/env bash
# please cd to SWSNet-main
pip install torch==2.1.1 torchvision==0.16.1 torchaudio==2.1.1 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
cd pointnet2_ops_lib
python setup.py install