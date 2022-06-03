import sys
import os
workPath = os.path.dirname(os.path.dirname(__file__))
if workPath not in sys.path:
    sys.path.append(workPath)

import torch
from config import INPUT_SIZE, NUM_CLASSFICATIONS,NUM_CELL_ANCHOR,BOX_ANCHORS

class ConvBL(torch.nn.Module):
    """ conv + batch norm + leaky
    """    
    def __init__(self,channel_in:int,channel_out:int,bias:bool=False,
                    kernel_size=(3,3),stride:int=1,padding:int=1,padding_mode='zeros'):
        super(ConvBL,self).__init__()

        self.cbl = torch.nn.Sequential(
            torch.nn.Conv2d(channel_in,channel_out,kernel_size=kernel_size,bias=bias,stride=stride,
                                            padding=padding,padding_mode=padding_mode),
            # 对不同通道的特征图同一位置（w,h）的上值进行 BN
            torch.nn.BatchNorm2d(channel_out),
            # 激活函数
            torch.nn.LeakyReLU(inplace=True)
        )

    def forward(self,input):
        """ 前向传播 """
        return self.cbl(input)

class UpSample(torch.nn.Module):
    """ 特征图上采样，图片放大一倍 """

    def __init__(self, scale_factor=2, mode="nearest"):
        super(UpSample, self).__init__()
        self.scale_factor = scale_factor
        self.mode = mode

    def forward(self, x):
        x = torch.nn.functional.interpolate(x, scale_factor=self.scale_factor, mode=self.mode)
        return x

class DownSample(torch.nn.Module):
    """ 利用卷积实现下采样，将图片缩小为一半 """
    def __init__(self,channel_in,channel_out,padding_mode='zeros') :
        super(DownSample,self).__init__()
        self.conv = torch.nn.Conv2d(channel_in,channel_out,(3,3),stride=2,bias=False,padding=1,padding_mode=padding_mode)

    def forward(self,input):
        return self.conv(input)

class Residual(torch.nn.Module):
    """ 残差模块 """
    def __init__(self,channel_in:int,cbl1_out:int,padding_mode='zeros'):
        super(Residual,self).__init__()

        self.cbl1 = ConvBL(channel_in,cbl1_out,kernel_size=(1,1),padding=0)
        self.cbl3 = ConvBL(cbl1_out,channel_in,padding_mode=padding_mode)

    def forward(self,input):
        x = self.cbl1(input)
        x = self.cbl3(x)
        return x + input

class ResX(torch.nn.Module):
    """ 整合 下采样 + x 次 Residual   """
    def __init__(self,channel_in:int,channel_out:int,repeetX:int,padding_mode='zeros'):
        super(ResX,self).__init__()

        # 降采样
        self.down = DownSample(channel_in,channel_out)

        # 重复 res 模型 x 次
        self.resx = torch.nn.ModuleList([])
        for i in range(repeetX):
            self.resx.append(Residual(channel_out, channel_in))

    def forward(self,input):
        # 下采样
        out = self.down(input)

        # 迭代残差模块
        for res in  self.resx:
            out = res(out)
        return out

class Backbone(torch.nn.Module):
    """ 主干网络 """ 

    def __init__(self):
        super(Backbone,self).__init__()

        # 3,416,416 ==> 32,416,416
        self.cbl = ConvBL(3, 32)

        # 32,416,416 ==> 64,208,208 x 1
        self.res1 = ResX(32, 64, 1)

        # 64,208,208 ==> 128,104,104 x 2
        self.res2 = ResX(64, 128, 2)

        # 128,104,104 ==> 256,52,52 x 8
        self.res8a = ResX(128, 256, 8)

        # 256,52,52 ==> 512,26,26 x 8
        self.res8b = ResX(256,512, 8)

        # 512,26,26 ==> 1024,13,13 x 4
        self.res4 = ResX(512,1024, 4)

    def forward(self,input):
        x = self.cbl(input)
        x = self.res1(x)
        x = self.res2(x)
        x52 = self.res8a(x)
        x26 = self.res8b(x52)
        x13 = self.res4(x26)
        return x52,x26,x13

class Conv5L(torch.nn.Module):
    """ conv 1x1 + conv 3x3 + conv 1x1 + conv 3x3 + conv 1x1 """
    def __init__(self,channel_in:int,channel_out:int,bias:bool=False,padding_mode='zeros'):
        super(Conv5L,self).__init__()
        self.conv5L = torch.nn.Sequential(
            ConvBL(channel_in,channel_out*2,kernel_size=(1,1),bias=bias,padding=0),
            ConvBL(channel_out*2,channel_out,kernel_size=(3,3),padding=1,padding_mode=padding_mode,bias=bias),
            ConvBL(channel_out,channel_out*2,kernel_size=(1,1),bias=bias,padding=0),
            ConvBL(channel_out*2,channel_out,kernel_size=(3,3),padding=1,padding_mode=padding_mode,bias=bias),
            ConvBL(channel_out,channel_out,kernel_size=(1,1),bias=bias,padding=0)
        )

    def forward(self,input):
        return self.conv5L(input)


class PredictLayers(torch.nn.Module):
    """ 根据主干网络的三层特征层，得到 yolo v3 的三层特征层输出 """
    def __init__(self,padding_mode='zeros',bias=False):
        super(PredictLayers,self).__init__()

        # 上采样
        self.up = UpSample()

        # ============ 13x13 特征 ================
        # 1024,13,13 ==> 1024,13,13
        self.conv5l13 = Conv5L(1024,1024)
        self.conv13 = torch.nn.Sequential(
            #  1024,13,13 ==> 512,13,13
            torch.nn.Conv2d(1024,512,(3,3),padding=1,bias=bias,padding_mode=padding_mode),
            # 512,13,13 ==> (1+4+NUM_CLASSFICATIONS)*3,13,13
            torch.nn.Conv2d(512,(5+NUM_CLASSFICATIONS)*NUM_CELL_ANCHOR,(1,1),padding_mode=padding_mode)
        )

        # ============ 26x26 特征 ================
        # 1024,13,13 ==> 256,13,13 
        self.conv126 = torch.nn.Conv2d(1024,256,(1,1),padding_mode=padding_mode)
        # up ==> 256,26,26
        # cat ==> 768,26,26
        # 768,26,26 ==> 256,26,26
        self.conv5l26 = Conv5L(768,256)
        self.conv26 = torch.nn.Sequential(
            #  256,26,26 ==> 128,26,26
            torch.nn.Conv2d(256,128,(3,3),padding=1,bias=bias,padding_mode=padding_mode),
            # 128,26,26 ==> (1+4+NUM_CLASSFICATIONS)*3,26,26
            torch.nn.Conv2d(128,(5+NUM_CLASSFICATIONS)*NUM_CELL_ANCHOR,(1,1),padding_mode=padding_mode)
        )

        # ============ 52x52 特征 ================
        # 256,26,26 ==> 128,26,26 
        self.conv152 = torch.nn.Conv2d(256,128,(1,1),padding_mode=padding_mode)
        # up ==> 128,52,52
        # cat ==> 384,52,52
        # 384,52,52 ==> 128,52,52
        self.conv5l52 = Conv5L(384,128)
        self.conv52 = torch.nn.Sequential(
            #  128,52,52 ==> 64,52,52
            torch.nn.Conv2d(128,64,(3,3),padding=1,bias=bias,padding_mode=padding_mode),
            # 64,52,52 ==> (1+4+NUM_CLASSFICATIONS)*3,52,52
            torch.nn.Conv2d(64,(5+NUM_CLASSFICATIONS)*NUM_CELL_ANCHOR,(1,1),padding_mode=padding_mode)
        )

    def forward(self,x52,x26,x13):
        # ============ 13x13 特征 ================
        x13 = self.conv5l13(x13)
        out13 = self.conv13(x13)

        # ============ 26x26 特征 ================
        x13 = self.conv126(x13)
        x13 = self.up(x13)
        #  256,26,26 + 512,26,26
        x26 = torch.cat([x13,x26],dim=1)
        x26 = self.conv5l26(x26)
        out26 = self.conv26(x26)

        # ============ 52x52 特征 ================
        x26 = self.conv152(x26)
        x26 = self.up(x26)
        #  128,52,52 + 256,52,52
        x52 = torch.cat([x26,x52],dim=1)
        x52 = self.conv5l52(x52)
        out52 = self.conv52(x52)


        out52 = self.reorganization(out52)
        out26 = self.reorganization(out26)
        out13 = self.reorganization(out13)

        return out52,out26,out13

    def reorganization(self,input):
        """将 ( batch, channel, height,width ) 转换为 ( bacth, anchor, height,width, [ tx,ty,tw,th ] + tc + classfication )
        并对 [ tx,ty,tw,th ] + tc + classfication  进行 sigmoid 处理

        Args:
            input (torch.Tensor): 模型预测结果

        Returns:
            tensor :  ( bacth, anchor, height,width, [ stx,sty,tw,th ] + stc + s_classfication )
        """        
        
        batch,_,height,width = input.shape

        # 将 ( batch, channel, height,width ) 转换为 ( bacth, anchor, height,width, [ tx,ty,tw,th ] + tc + classfication )
        out = input.reshape((batch,NUM_CELL_ANCHOR,(5 + NUM_CLASSFICATIONS), height,width)).permute(0,1,3,4,2)

        # 对 [ tx,ty,tw,th ] + tc + classfication 添加 sigmoid
        # sigmoid([tx,ty])
        out[...,0:2] = torch.sigmoid(out[...,0:2])

        # sigmoid([ tc, classfication ])
        out[...,4:] = torch.sigmoid(out[...,4:])

        return out 


class Darknet53(torch.nn.Module):
    """ darknet 53 输出结果为：
       ( bacth, anchor, height,width, [ stx,sty,tw,th ] + stc + s_classfication )
    
    """
    def __init__(self,padding_mode='zeros',bias=False):
        super(Darknet53,self).__init__()
        self.backbone = Backbone()
        self.predictLayers =  PredictLayers()

    def forward(self,input):
        outs = self.backbone(input)
        return self.predictLayers(*outs)


if __name__ == '__main__':
    # 完整网络测试
    # darknet = Darknet53()
    # img = torch.ones((2,3,416,416))
    # y1,y2,y3 = darknet(img)

    # 预测结果重组测试
    img = torch.ones((2,3*7,5,5))
    img = img + torch.linspace(0,4,5).reshape(1,5) 
    predict = PredictLayers()
    predict.reorganize(img,torch.Tensor([[1,2],[3,4],[5,6]]))
    


