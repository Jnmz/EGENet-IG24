# EGENet

## 1. Environment setup

```
Python 3.6
pytorch 1.6.0
torchvision 0.7.0
einops  0.3.0
timm
```

## 2. Download the datesets:
* LEVIR-CD:
[LEVIR-CD](https://justchenhao.github.io/LEVIR/)
* WHU-CD:
[WHU-CD](https://study.rsgis.whu.edu.cn/pages/download/building_dataset.html)

and put them into data directory. In addition, the processed whu dataset can be found in the release.

## 3. Download the models (pretrain models):

* [resnet18](https://download.pytorch.org/models/resnet18-5c106cde.pth) 

and put it into pretrain directory.

## 4. Train & Test

    python main_cd.py
    
    
    
