# %% 
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
                                            transform=toTensor,download=False)
testImages = torchvision.datasets.CIFAR10('../asset/cifa10',train=False,
                                            transform=toTensor,download=False)
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

# %% 测试

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
