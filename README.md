# 1. MER based on GCL for Senior
This is the implementation module for multimodal emotion recognition (MER) based on graph contrastive learning (GCL) on Senior empirical dataset
The database consists of text and audio which is acquired when uttering scripts evoking certain emotions.
There are 7 target emotions: 1) joy, 2) neutral, 3) anxiety, 4) embarrassment, 5) hurt, 6) sadness, and 7) anger

## Dependencies
* torch
* pandas 
* numpy
* sklearn
* pyyaml
* typing
* matplotlib
* datetime

Install all dependencies as

'''
pip install -r requirements.txt
'''

You should consider deep learning setups such as CUDA and PyTorch versions available in your local environments.

## Usage

Train and evaluate the model by executing as

'''
python train.py --dataset IITP-SMED
'''

Available --dataset arguments must be one of [IITP-SMED, IITP-SMED-STT, AIHUB-SER]
IITP-SMED and IITP-SMED-STT are our empirical datasets constructed by taking funds from IITP in South Korea.
See details of AIHUB-SER datasets online available [link](https://www.aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&dataSetSn=263). 
