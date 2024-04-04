# EGENet

## 1. Environment setup

```
CUDA 11.1
Python 3.8
pytorch 1.8.1
torchvision 0.9.1
einops 0.7.0
timm 0.9.16
```

## 2. Download the datesets:
* LEVIR-CD:
[LEVIR-CD](https://pan.baidu.com/s/1gDS6Ea37zfHoZ4832jT9cg?pwd=BUPT)
* WHU-CD:
[WHU-CD](https://github.com/Jnmz/EGENet-IG24/releases/download/1/WHU256.zip)

and put them into data directory. In addition, the processed whu dataset can be found in the release.

## 3. Download the models (pretrain models):

* [resnet18](https://download.pytorch.org/models/resnet18-5c106cde.pth) 

and put it into pretrain directory.

## 4. Train & Test

    python main_cd.py --project_name 'EGENet_LEVIR' --data_name 'LEVIR' --net_G 'EGENet'
    python main_cd.py --project_name 'EGENet_WHU' --data_name 'WHU' --net_G 'EGENet'
    python main_cd.py --project_name 'EGCTNet_LEVIR' --data_name 'LEVIR' --net_G 'EGCTNet'
    python main_cd.py --project_name 'EGCTNet_WHU' --data_name 'WHU' --net_G 'EGCTNet'
    python main_cd.py --project_name 'BIT_LEVIR' --data_name 'LEVIR' --net_G 'BIT' --loss 'ce'
    python main_cd.py --project_name 'BIT_WHU' --data_name 'WHU' --net_G 'BIT' --loss 'ce'
    python main_cd.py --project_name 'ICIF_Net_LEVIR' --data_name 'LEVIR' --net_G 'ICIF_Net' --loss 'ce'
    python main_cd.py --project_name 'ICIF_Net_WHU' --data_name 'WHU' --net_G 'ICIF_Net' --loss 'ce'
    python main_cd.py --project_name 'ChangeFormer_LEVIR' --data_name 'LEVIR' --net_G 'ChangeFormer' --loss 'ce'
    python main_cd.py --project_name 'ChangeFormer_WHU' --data_name 'WHU' --net_G 'ChangeFormer' --loss 'ce'
    
    
