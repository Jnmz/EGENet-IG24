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
[WHU-CD](https://study.rsgis.whu.edu.cn/pages/download/building_dataset.html)

and put them into data directory. In addition, the processed whu dataset can be found in the release.

## 3. Download the models (pretrain models):

* [resnet18](https://download.pytorch.org/models/resnet18-5c106cde.pth) 

and put it into pretrain directory.

## 4. Train & Test

    python main_cd.py
    
    
    
