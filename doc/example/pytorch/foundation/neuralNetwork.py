# %% 
import torch

# %% 框架

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

# %% sequential

class Model(torch.nn.Module):
    def __init__(self):
        """ 初始化网络层 """
        # 初始化父类
        super(Model, self).__init__()
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

# 创建模型
model = Model()

# 模型前向传播
x = torch.ones((3,3,64,64))

# 模型输出
out = model(x)
print(out.shape)

