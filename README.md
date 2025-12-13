
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
本repo是论文《遥感图像指示分割的基准数据集与基线模型研究》的实现代码及相关数据集介绍。

## 将要做的
- [x] 公布部分数据集
- [x] 公布检查点用于推理
- [ ] 公布所有数据集及训练代码

## 安装
* Ubantu==22.04
* Python==3.7 
* Torch==1.10.1, Torchvision==0.11.0
* CUDA==11.1
* 检查点==[RSIRS](https://drive.google.com/file/d/1auMd9sOpYcAaIelJVKPOKvBYKic7yy4w/view?usp=drive_link)
* 部分数据集==[PART_DATASET](https://drive.google.com/file/d/1a-U9jg_xd2BDMQk8Fk8hJqYBOYS9iv01/view?usp=drive_link)
* Bert（语言预训练模型）==[Bert](https://drive.google.com/file/d/1ee-XVDnqTNj3tBqgc1S2WLFEMY2dv2iU/view?usp=drive_link)


```
conda create -n rsirs python=3.7
conda activate rsirs 
pip install torch==1.10.0+cu111 torchvision==0.11.0+cu111 torchaudio==0.10.0 -f https://download.pytorch.org/whl/torch_stable.html
```

```
cd /RSIRS
pip install -r requirments.txt
```
## 推理
* 从您下载的部分数据集中选取一张图片，获得其绝对路径例如/home/test/1.png
* 查看图片，选择您需要分割的目标，根据目标在图像中的位置或其本身的颜色信息指定输入文本。例如对一辆红色的车可如下描述：find a red car on the road
* 打开主目录下的args文件，将图像路径填入--infer_img_path中，将语言填入--infer_language中，并在--resume中填入您下载的检查点路径，在--infer_img_savepath中输入您期望保存的图片路径，请使用.png后缀
* 按如下代码运行得到推理结果
```
python inference.py --img_path xxx.img
```
## 可能的问题
* 如果您不熟悉该领域，您可能在配置bert时遇到错误，需要手动设置本地bert路径解决该问题。 如您遇到问题，请在本repo中提问，我们将协助您解决。
