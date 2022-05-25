# 神经网络

# 1. 网络层


## 1.1. 网络框架

1. 继承神经网络模型类 `torch.nn.Module`
2. 在 `__init__` 中，初始化父类并创建「网络模块」
3. 重写 `forward` ，组织「网络模块」

```python
import torch

class Model(torch.nn.Module):
    def __init__(self):
        """ 初始化网络层 """
        # 初始化父类
        super(Model, self).__init__()
        self.fcLyaer1 = torch.nn.Linear(32,16)
        self.fcLyaer2 = torch.nn.Linear(16,8)

    def forward(self,input):
        """ 搭建网络 """
        a1 = self.fcLyaer1(input)
        a2 = self.fcLyaer2(a1)
        return a2

# 创建模型
model = Model()


# 模型前向传播
x = torch.ones((3,32))

# 模型输出
out = model(x)
```
- `model.train()`：启动模型训练模式，在该模式下，例如 drop out 这些特殊层，才会被执行
- `model.eval()`：启动模型预测模式，该模式会关闭辅助训练特殊模块，例如 drop out

## 1.2. 网络模块

- <a href="https://pytorch.org/docs/stable/nn.html" class="jump_link"> 网络模块 </a>

### 1.2.1. 全连接层

- **输入：** (N, in_features)，N 为输入样本数
- **输出：** (N, out_features)

```python
# in_features：输入的维度
# out_featuers：输出的维度
# bias：是否启用 b
torch.nn.Linear(in_features : int, out_features: int, bias: bool=True)

# 根据输入自动判断 in_features
torch.nn.LazyLinear( out_features: int, bias: bool=True)
```

### 1.2.2. 二维卷积

- **输入：** $(N,C_{in},H,W)$
- **输出：** $(N,C_{out},H_{out},W_{out})$

    $$
    \begin{aligned}
    H_{\rm {out }} &=\left[\frac{H_{i n}+2 \times \rm { padding }[0]-\rm { dilation }[0] \times(\rm { kernel\_size }[0]-1)-1}{\operatorname{stride}[0]}+1\right]\\
    W_{\rm {out }} &=\left[ \frac{W_{\rm {in }}+2 \times \rm { padding }[1]-\operatorname{dilation}[1] \times(\rm { kernel\_size }[1]-1)-1}{\rm { stride }[1]}+1 \right ]
    \end{aligned}
    $$

- **dilation：** 卷积核膨胀
    <p style="text-align:center;"><img src="/artificial_intelligence/image/neuralNetwork/dilation.gif" width="25%" align="middle" /></p>

- <a href="https://github.com/vdumoulin/conv_arithmetic/blob/master/README.md" class="jump_link"> 每个参数的意义 </a>

```python
# in_channels：输入图片的通道
# out_channels：输出图片的通道
# kernel_size：卷积核的尺寸
# stride：卷积移动步长
# dilation：卷积核膨胀倍数
# groups：卷积核的分配，1 就是常规意义的卷积
torch.nn.Conv2d(in_channels: int, out_channels: int, 
                kernel_size, stride=1, 
                padding=0, , padding_mode: str='zeros'
                dilation=1, groups: int=1, bias: bool=True)
```

### 1.2.3. 三维卷积

<p style="text-align:center;"><img src="/artificial_intelligence/image/pytorch/conv3.png" width="25%" align="middle" /></p>

- **输入：** $(N,C_{in},D,H,W)$，在二维图片的基础上多了一个维度，用来表示三维的数据。
- **输出：** $(N,C_{out},D_{out},H_{out},W_{out})$，经过卷积后的结果是三维的
- **卷积核：** 卷积核是三维的小立方体，在三维数据的三个方向上滑动。


```python
torch.nn.conv3d(in_channels: int, out_channels: int, 
                kernel_size: _size_2_t, stride: _size_2_t=1, 
                padding=0, , padding_mode: str='zeros'
                dilation=1, groups: int=1, bias: bool=True)
```

### 1.2.4. 池化层

- **ceil_mode**：当池化层的卷积核出界时，结果是否保留
    <p style="text-align:center;"><img src="/artificial_intelligence/image/pytorch/pool_ceil.png" width="25%" align="middle" /></p>

```python
# 最大池化
torch.nn.MaxPool2d(kernel_size, stride, padding, dilation=1, 
                    ceil_mode=False)

# 平均池化层
torch.nn.AvgPool2d(kernel_size, stride, padding, dilation=1, 
                    ceil_mode=False)
```

### 1.2.5. 激活层

- <a href="https://pytorch.org/docs/stable/nn.html#non-linear-activations-weighted-sum-nonlinearity" class="jump_link"> 激活函数 </a>
- **inplace：** 计算结果是否覆盖原来的输入
```python
torch.nn.ReLU(inplace=False)
```


### 1.2.6. 其他层

- <a href="https://pytorch.org/docs/stable/generated/torch.nn.BatchNorm2d.html#torch.nn.BatchNorm2d" class="jump_link"> batch norm </a>
- <a href="https://pytorch.org/docs/stable/nn.html#dropout-layers" class="jump_link">  Drop out</a>

## 1.3. sequential

- **作用**：将多个神经网络层组合为一个

```python
class Model(torch.nn.Module):
    def __init__(self):
        """ 初始化网络层 """
        # 初始化父类
        super(Model, self).__init__()

        # 将多模块整合
        self.sequentialNet = torch.nn.Sequential(
            torch.nn.Conv2d(3,4,(3,3)),
            torch.nn.MaxPool2d((2,2)),
            torch.nn.ReLU(True),
            torch.nn.Flatten(),
            torch.nn.Linear(3844,16),
            torch.nn.ReLU(True),
            torch.nn.Linear(16,8)
        )

    def forward(self,input):
        """ 搭建网络 """
        out = self.sequentialNet(input)
        return out
```

## 1.4. 损失函数

- <a href="https://pytorch.org/docs/stable/generated/torch.nn.L1Loss.html#torch.nn.L1Loss" class="jump_link"> 平均绝对误差 MAE（mean absolute error） </a> 
   ```python
   torch.nn.L1Loss(size_average:bool, reduce:bool, reduction='mean')
   ```

- <a href="https://pytorch.org/docs/stable/generated/torch.nn.MSELoss.html#torch.nn.MSELoss" class="jump_link"> 均方差 MSE </a>
   ```python
   torch.nn.MSELoss(size_average:bool, reduce:bool, reduction='mean')
   ```
- <a href="https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html#torch.nn.CrossEntropyLoss" class="jump_link"> 交叉熵 </a>：会根据标签是「顺序编码」还是「独热编码」来选择对应的交叉熵损失函数
   ```python
   orch.nn.CrossEntropyLoss(weight=None, size_average:bool, ignore_index=- 100, reduce:bool, reduction='mean', label_smoothing=0.0)
   ```

## 1.5. 经典网络模型

- <a href="https://pytorch.org/vision/stable/models.html#classification" class="jump_link"> 机器视觉经典模型 </a>
- <a href="https://pytorch.org/audio/stable/models.html" class="jump_link"> 语音处理模型 </a>

```python
# vgg16 模型
# pretrained：模型的系数是否重新训练
torchvision.models.vgg16(pretrained: bool = False, progress: bool = True, **kwargs: Any)
```

# 2. 优化器

- <a href="https://pytorch.org/docs/stable/optim.html" class="jump_link"> 神经网络优化算法 </a>

```python
# adam 算法
# params：神经网络的相关系数，例如 w，b，卷积核等
optimizer = torch.optim.Adam(params=net.parameters(), lr=0.001, 
        betas=(0.9, 0.999), eps=1e-08, weight_decay=0, 
        amsgrad=False, *, maximize=False)

# 相关系数梯度重置
optimizer.zero_grad()

# 反向传播，根据损失函数，计算系数梯度
loss.backward()

# 相关系数在反向传播后进行一次更新
optimizer.step()
```

> [!note] 
> 在「反向传播」之前，一定要把系数的梯度进行重置。否则再次反向传播后，梯度计算会出问题。

# 3. 模型保存与读取

- **保存模型结构信息+系数**
    ```python
    # 保存模型，文件后缀为 .pth 
    torch.save(net,path)

    # 加载模型
    torch.load(path)
    ```
- **保存模型系数**
   ```python
   # 保存系数，文件后缀为 .pth
   torch.save(net.state_dict(),path)

   # 读取
   net.load_state_dict(torch.load(path)) 
   ```

> [!note|style:flat]
> 方案一保存模型，仅仅只是保存了模型信息，模型的 `class` 并没有保存，因此，导入模型时，还是需要给出模型的具体 `class`


# 4. GPU 调用

## 4.1. GPU 训练

- **思路：** 网络模型、数据（样本与标签）、损失函数全部添加到GPU中
- 实现：

    -  `to()`
  
        ```python
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # 模型
        model = Model()
        model = model.to(device)

        # 损失函数
        lossFcn = torch.nn.CrossEntropyLoss()
        lossFcn = lossFcn.to(device)

        # 数据
        datas = datas.to(device)
        ```
    
    - `cuda()`

        ```python
        # 模型
        model = Model()
        model = model.cuda()

        # 损失函数
        lossFcn = torch.nn.CrossEntropyLoss()
        lossFcn = lossFcn.cuda()

        # 数据
        datas = datas.cuda()
        ```

## 4.2. 加载模型

**对「GPU」训练的模型，想要在「CPU」上进行测试时，需要在加载时进行转换**
```python
model = torch.load('../model/cifar_19.pth',map_location=torch.device('cpu'))
```


# 附录：cifa10 案例

<p style="text-align:center;"><img src="/artificial_intelligence/image/pytorch/cifar10Model.png" width="75%" align="middle" /></p>

```python
import torch
import torchvision
from  torch.utils.tensorboard import SummaryWriter
from PIL import Image
# %% 模型
class Model(torch.nn.Module):
    """ 分类模型 """
    def __init__(self):
        super(Model, self).__init__()
        self.cifar = torch.nn.Sequential(
            torch.nn.Conv2d(3,6,(5,5)),
            torch.nn.MaxPool2d(2),
            torch.nn.ReLU(True),
            torch.nn.Conv2d(6,16,(5,5)),
            torch.nn.MaxPool2d(2),
            torch.nn.ReLU(True),
            torch.nn.Flatten(),
            torch.nn.Linear(400,120),
            torch.nn.ReLU(True),
            torch.nn.Linear(120,84),
            torch.nn.ReLU(True),
            torch.nn.Linear(84,10)
        )

    def forward(self,input):
        out = self.cifar(input)
        return out

# %% 训练

# 设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 日志
logWriter = SummaryWriter('../logs')

# 转换图片为 tensor
toTensor = torchvision.transforms.ToTensor()

# 数据集
trainImages = torchvision.datasets.CIFAR10('../asset/cifa10',train=True,
                                            transform=toTensor,download=True)
testImages = torchvision.datasets.CIFAR10('../asset/cifa10',train=False,
                                            transform=toTensor,download=True)
# loader
trainLoader = torch.utils.data.DataLoader(trainImages,64)
testLoader = torch.utils.data.DataLoader(testImages,128)

# 参数
epochs = 20
learnRate = 0.001

# 模型
cifarNet = Model()
cifarNet = cifarNet.to(device)


# 损失函数
lossFcn = torch.nn.CrossEntropyLoss()
lossFcn = lossFcn.to(device)

# 优化器
optimizer = torch.optim.Adam(cifarNet.parameters(),lr=learnRate)

for epoch in range(epochs):
    print("--------------epoch {} ----------------".format(epoch))
    batchCount = 0
    lossSum = 0.0

    # 训练
    for batch in trainLoader:
        # 获取训练数据与标签
        datas,targets = batch
        datas = datas.to(device)
        targets = targets.to(device)
        
        # 前向传播
        cifarNet.train()
        out = cifarNet(datas)


        # 计算损失
        loss = lossFcn(out,targets) 
        lossSum = lossSum + loss.item()
        
        # 重置梯度
        optimizer.zero_grad()

        # 反向传播
        loss.backward()

        # 更新梯度
        optimizer.step()

        # 打印损失
        batchCount = batchCount + 1
        if batchCount % 100 == 0:
            print("\tbatch: {},loss: {}".format(batchCount,loss.item()))

    print("epoch: {},loss: {}".format(epoch,lossSum / len(trainLoader)))
    # 记录损失
    logWriter.add_scalar('train loss', lossSum / len(trainLoader),global_step=epoch)

    # 测试
    rightCount = 0
    for batch in testLoader:
        datas,targets = batch
        datas = datas.to(device)
        targets = targets.to(device)

        # 关闭梯度计算
        with torch.no_grad():
            # 预测
            cifarNet.eval()
            out = cifarNet(datas)
    
            # 得到分类
            predict = torch.argmax(out,dim=1)
            
            # 记录分类正确的
            rightCount = rightCount + (predict==targets).sum()

    print("epoch: {},accuracy: {}".format(epoch,float(rightCount) / len(testImages)))
    # 记录损失
    logWriter.add_scalar('test accuracy', float(rightCount) / len(testImages),global_step=epoch)

    # 保存模型
    torch.save(cifarNet,'../model/cifar_{}.pth'.format(epoch))

logWriter.close()
```

**模型测试**

```python
resize = torchvision.transforms.Resize((32,32))
# 预测图片
img = Image.open('../asset/cat.jpeg')
img = resize(img)
imgTensor = toTensor(img).reshape(1,3,32,32)

# 加载模型
cifarNet = torch.load('../model/cifar_19.pth',map_location=torch.device('cpu'))

# 关闭梯度计算
with torch.no_grad():
    # 预测
    cifarNet.eval()
    out = cifarNet(imgTensor)

    # 得到分类
    predict = torch.argmax(out,dim=1)
    print(testImages.classes[predict])
```