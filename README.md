
<p align="center">
  <h1 align="center">遥感图像指示分割的基准数据集与基线模型研究</h1>
  <p align="center">


   <br />
    <strong>赵显宇</strong></a>
    ·
    <strong>姜会林</strong></a>
    ·
    <strong>王合龙</strong></a>
    ·
    <strong>李萌</strong></a>    
    ·
    <strong>赵琦</strong></a>
    ·
    <strong>刘智</strong></a>
    ·
    <strong>赵克军</strong></a>
    <br />
<p align="center">

    
  </p>





## 重点!!!!
本repo是论文《遥感图像指示分割的基准数据集与基线模型研究》的实现代码及相关数据集介绍。我们参考了[RSMIN](https://github.com/Lsan2401/RMSIN)，非常感谢这篇优秀的工作。

## 将要做的
- [x] 公布部分数据集
- [x] 公布检查点用于推理
- [ ] 公布所有数据集及训练代码

## 安装
* Ubantu==18.04
* Python==3.10 
* Torch==1.12.1, Torchvision==0.12.0
* CUDA==11.3
* checkpoint==[LM2CNet](https://drive.google.com/file/d/1auMd9sOpYcAaIelJVKPOKvBYKic7yy4w/view?usp=drive_link)
* Test_Dataset==[Test_Dataset](https://drive.google.com/file/d/1a-U9jg_xd2BDMQk8Fk8hJqYBOYS9iv01/view?usp=drive_link)
* Bert_Pretrain==[Bert](https://drive.google.com/file/d/1ee-XVDnqTNj3tBqgc1S2WLFEMY2dv2iU/view?usp=drive_link)


**Please add the Bert_Pretrain into ./roberta-base folder**


**Please add the checkpoint into ./config folder**
```
conda create -n LM2CNet python=3.10
conda activate LM2CNet
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113
```

```
cd /LM2CNet
pip install -r requirments.txt
```

```
cd /LM2CNet/lib/models/LM2CNet/ops
bash make.sh
```
## Test on Mono3DRefer_nuScenes
**please set the Test_Dataset path**


**please set the checkpoint path**
```
cd /LM2CNet
python test.py
```
## Demo Video

[Demo](https://github.com/user-attachments/assets/9e6e3e33-5ebb-4dd2-9f3f-83f46848e5e6)

## Dataset Construction
**If you want to use chatgpt to automatically generate language description, we have provided a demo tool in folder /LM2CNet/datasets_construction for your reference.**


**If you want to use DriveLM to indentify the nuScens object, we have provided a demo tool in folder /LM2CNet/datasets_construction for your reference.**

## Others
**If you have any problems, please contact me limenglm@buaa.edu.cn**
