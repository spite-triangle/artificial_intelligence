# 图片处理

# 基础概念
## 模块介绍

在进行计算机视觉相关模型搭建时，需要对「图片」这类数据进行许多的处理，PyTorch 提供了一些基础的图片处理工具。负责的图片处理还是得靠 <a href="https://spite-triangle.github.io/artificial_intelligence/#/./ComputerVision/chapter/README" class="jump_link"> OpenCV </a> 。

```python
from  torchvision import  transforms
```

## Transform 实现

对于 `transforms` 模块中的变换操作均是定义为了「类」而非具体函数，但是该类并没有提供「方法」实现操作调用，而是通过重定义符号`()`，即重写方法`__call__`，来执行具体的操作逻辑，例如 `ToTensor` 类

```python
class ToTensor:
    def __init__(self) -> None:
        _log_api_usage_once(self)

    def __call__(self, pic):
        return F.to_tensor(pic)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"
```

**因此，Transforms 的实现就是利用「类」伪装成「函数」。**

## Transform 调用

**步骤**：
1. 实例化`transforms`中的操作类
2. 通过 `()` 实现具体操作逻辑调用

```python
from  torchvision import  transforms
from PIL import  Image

# 读取图片
img = Image.open('../asset/cat.jpeg')

# 实例化 transform
toTensor = transforms.ToTensor()

# 对图片进行 transform
imgTensor = toTensor(img)
```
## 图片数据

Transforms 能识别的图片数据类型为
- **PIL Image**：通道顺序为 (H x W x C)
- **numpy.ndarray**：通道顺序为 (H x W x C)，<span style="color:red;font-weight:bold"> 对于 OpenCV 而言，还需要把蹩脚的 BGR 通道转换为 RGB </span>

PyTorch 进行网络训练时，一般使用的图片数据类型为

- **Tensor**： 通道顺序 (C x H x W) ，且颜色通道值被归一化到 `[0,1]`


# Transforms

## 常用 transforms

```python
# 是转换数据格式，把数据转换为tensfroms格式。只有转换为tensfroms格式才能进行后面的处理。
transforms.ToPILImage() 

# 是按照比例把图像最小的一个边长放缩到256，另一边按照相同比例放缩。
transforms.Resize(256)

# 是把图像按照中心随机切割成224正方形大小的图片。
transforms.RandomResizedCrop(224,scale=(0.5,1.0))

# 转换为tensor格式，这个格式可以直接输入进神经网络了。
transforms.ToTensor() 

# 对像素值进行归一化处理。
transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
```

## 其他 transforms

- <a href="https://blog.csdn.net/u011995719/article/details/85107009" class="jump_link"> transforms 的二十二个方法 </a>

- **裁剪**

```python
# 中心裁剪：
transforms.CenterCrop()

# 随机裁剪：
transforms.RandomCrop()

# 随机长宽比裁剪：
transforms.RandomResizedCrop()

# 上下左右中心裁剪：
transforms.FiveCrop()

# 上下左右中心裁剪后翻转，
transforms.TenCrop()
```
- **旋转和翻转**

```python
# 依概率p水平翻转：
transforms.RandomHorizontalFlip(p=0.5)

# 依概率p垂直翻转：
transforms.RandomVerticalFlip(p=0.5)

# 随机旋转：
transforms.RandomRotation()
```
- **图形变换**

```python
# resize：
transforms.Resize()

# 标准化：
transforms.Normalize()

# 转为tensor，并归一化至[0-1]：
transforms.ToTensor()

# 填充：
transforms.Pad()

# 修改亮度、对比度和饱和度：
transforms.ColorJitter()

# 转灰度图：
transforms.Grayscale()

# 线性变换：
transforms.LinearTransformation()

# 仿射变换：
transforms.RandomAffine()

# 依概率p转为灰度图：
transforms.RandomGrayscale()

# 将数据转换为PILImage：
transforms.ToPILImage()
```

## transforms 批处理

```python
img = Image.open('../asset/cat.jpeg')

# 将多个 transform 组合成一个操作
compose = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Resize(512),
        transforms.Grayscale(),
        transforms.ToPILImage()
    ]
)

img = compose(img)
img.show()
```
