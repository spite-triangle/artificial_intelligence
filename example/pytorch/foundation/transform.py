# %%
from  torchvision import  transforms,datasets
from PIL import  Image
from torch.utils.tensorboard import SummaryWriter
import torch
import  numpy as np
import  os
# %% 图片导入

img = Image.open('../asset/cat.jpeg')

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

# %% 标准数据集

cifar10 = datasets.CIFAR10('../asset/cifar10',train=True,download=True)

# %% 自定义数据集

class ImgaeAssets(torch.utils.data.Dataset):
    def __init__(self,path):
        self.root = path
        self.files = os.listdir(path)
        pass

    def __getitem__(self,id):
        """ 用于数据集中的样本获取 """
        filePath = os.path.join(self.root,self.files[id])
        img = Image.open(filePath)
        return img

    def __len__(self):
        """ 数据的数量 """
        return len(self.files)

# 创建数据集
assets = ImgaeAssets('../asset')

# 获取数据
img = assets[0]
img.show()

# %% tensorboard

toTensor = transforms.ToTensor()

# 创建日志生成器
writer = SummaryWriter('../log')

# 添加日志内容
writer.add_image('cat', toTensor(assets[0]))

x = torch.linspace(-10, 10,100,dtype=torch.float)

for i in range(len(x)):
    writer.add_scalar('sin', torch.sin(x[i]),global_step=i)


# 关闭
writer.close()

# %% dataloader

loader = torch.utils.data.DataLoader(dataset=assets,batch_size=2,shuffle=False,drop_last=False)

