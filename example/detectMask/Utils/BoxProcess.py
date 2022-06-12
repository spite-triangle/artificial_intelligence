from logging import config
import sys
import os
workPath = os.path.dirname(os.path.dirname(__file__))
if workPath not in sys.path:
    sys.path.append(workPath)

from xml.dom.minidom import parse
import torch
from config import OBJECT_CLASSFICATIONS,INPUT_SIZE,NUM_CELL_ANCHOR,RUN_DEVICE
import numpy as np



def predictionDecode(prediction:torch.Tensor,anchorBoxs:torch.Tensor) -> torch.Tensor:
    """将属性 [stx,sty,tw,th,stc] 值进行转换为 [bx,by,bw,bh,bc,b_classfication]，并将结果尺寸映射到 INPUT_SIZE

    Args:
        prediction (torch.Tensor): yolo 网络模型计算结果， ( bacth, anchor, height,width, [ stx,sty,tw,th ] + stc + sclassfication )
        anchorBoxs (torch.Tensor): [[ anchor_w,anchor_h ]] 
    Returns:
        out (torch.Tensor):  ( batch, anchor, height,width, [ bx,by,bw,bh ] + bc + b_classfication ) ，
                            [ bx,by,bw,bh ] 的大小基于 INPUT_SIZE
    """       
    _,_, height,width, _ = prediction.shape

    # 计算感受野
    receptionField = INPUT_SIZE[0] / width

    # bx = ( sigmoid(tx) + Cx ) * receptionFeild
    prediction[...,0] = ( prediction[...,0] + torch.linspace(0, width - 1, width,device=RUN_DEVICE).reshape(1,width) ) * receptionField

    # by = ( sigmoid(ty) + Cy ) * receptionFeild
    prediction[...,1] = ( prediction[...,1] + torch.linspace(0,height - 1,height,device=RUN_DEVICE).reshape(height,1) )  * receptionField

    # bw = Pw * exp(tw)
    # torch.exp_(prediction[...,2])：计算结果的形状为 ( bacth,anchor,height,width )
    # permute(0,2,3,1)：调整形状为 ( bacth, height,width,anchor )  这样方便对 anchor 维度进行广播
    # banchors[:,0].reshape(1,NUM_CELL_ANCHOR) ：形状为 ( 1, anchor ) 值就为 [[w1,w2,w3]]
    # permute(0,3,1,2): 将 ( bacth, height,width,anchor ) 还原为 ( bacth,anchor, height,width ) 
    prediction[...,2] = ( torch.exp(prediction[...,2]).permute(0,2,3,1) * anchorBoxs[:,0].reshape(1,NUM_CELL_ANCHOR)).permute(0,3,1,2)

    # bh = Ph * exp(th) 
    # torch.exp_(prediction[...,2])：计算结果的形状为 ( bacth,anchor, height,width )
    # permute(0,2,3,1)：调整形状为 ( bacth, height,width,anchor )  这样方便对 anchor 维度进行广播
    # banchors[:,1].reshape(1,NUM_CELL_ANCHOR) ：形状为 ( 1, anchor ) 值就为 [[h1,h2,h3]]
    # permute(0,3,1,2): 将 ( bacth, height,width,anchor ) 还原为 ( bacth,anchor, height,width ) 
    prediction[...,3] = ( torch.exp(prediction[...,3]).permute(0,2,3,1) * anchorBoxs[:,1].reshape(1,NUM_CELL_ANCHOR) ).permute(0,3,1,2)

    # bc + b_classfication = sigmoid(tc + classfication)
    # prediction[...,4:] = prediction[...,4:]
    return prediction



def addPaddingToBoxs(boxs: torch.Tensor, padding: torch.Tensor, scale: float = 1) -> torch.Tensor:
    """在原始标签boxs的基础上，添加 padding

    Args:
        boxs (torch.Tensor): 外接矩形，[ [x1,y1,x2,y2] ]
        padding (torch.Tensor): 将图片变为正方形，在  [ 上下(y)。左右(x) ] 填充的像素
        scale (float())：缩放比列

    Returns:
        torch.Tensor: 修正后的boxs
    """
    boxs = boxs + torch.tensor([padding[1], padding[0],
                           padding[1], padding[0]])
    return (boxs * scale).type(torch.int)

def addPaddingToLable(label:torch.Tensor, padding: torch.Tensor, scale: float = 1) -> torch.Tensor:
    """对标签进行padding修正

    Args:
        label (torch.Tensor): [[图片编号,分类,x1,y1,x2,y2]]
        padding (torch.Tensor): 将图片变为正方形，在  [ 上下(y)。左右(x) ] 填充的像素
        scale (float, optional): 填充 padding 后，INPUT_SIZE / 原始图片尺寸. Defaults to 1.

    Returns:
        torch.Tensor: 添加padding后的label
    """
    label[:,2:6] =  ( label[:,2:6] + torch.tensor([ padding[1], padding[0],
                           padding[1], padding[0] ]) ) * scale
    return label

def getLabelFromXml(xmlPath) -> torch.Tensor:
    """从 xml 标签文件中获取 label  [[图片编号,分类,x1,y1,x2,y2]]

    Args:
        xmlPath (str): 标签路径

    Returns:
        torch.Tensor: [[图片编号,分类,x1,y1,x2,y2]]
    """

    dom = parse(xmlPath)
    boxElements = dom.documentElement.getElementsByTagName('bndbox')
    # 储存坐标
    boxs = []
    for boxElement in boxElements:
        objectName =  boxElement.parentNode.getElementsByTagName('name')[0].childNodes[0].nodeValue
        objectClass =  OBJECT_CLASSFICATIONS[objectName] 
        x1 = boxElement.getElementsByTagName('xmin')[0].childNodes[0].nodeValue
        y1 = boxElement.getElementsByTagName('ymin')[0].childNodes[0].nodeValue
        x2 = boxElement.getElementsByTagName('xmax')[0].childNodes[0].nodeValue
        y2 = boxElement.getElementsByTagName('ymax')[0].childNodes[0].nodeValue
        boxs.append([0,objectClass, int(x1), int(y1), int(x2), int(y2)])

    return torch.tensor(boxs,dtype=torch.float)


def box_xy2cwh(boxs):
    """将 bounding box 由 [[ x1,y1,x2,y2 ]] 更改为 [[ xc,yc,w,h ]]

    Args:
        boxs (np.ndarray,torch.Tensor):  [[ x1,y1,x2,y2 ]] 形式的 bounding box
    """
    # 类型判断
    if isinstance(boxs, torch.Tensor):
        out = torch.zeros_like(boxs,dtype=torch.float32)
    elif isinstance(boxs, np.ndarray):
        out = np.zeros_like(boxs,dtype=np.float32)

    out[:,0] = (boxs[:,0] + boxs[:,2]) * 0.5
    out[:,1] = (boxs[:,1] + boxs[:,3]) * 0.5
    out[:,2] = (boxs[:,2] - boxs[:,0]) 
    out[:,3] = (boxs[:,3] - boxs[:,1]) 

    return out

def box_cwh2xy(boxs):
    """将 bounding box 由 [[ xc,yc,w,h ]] 更改为 [[ x1,y1,x2,y2 ]]

    Args:
        boxs (np.ndarray,torch.Tensor):  [[ xc,yc,w,h ]] 形式的 bounding box
    """
    # 类型判断
    if isinstance(boxs, torch.Tensor):
        out = torch.zeros_like(boxs)
    elif isinstance(boxs, np.ndarray):
        out = np.zeros_like(boxs)

    out[:,0] = boxs[:,0] - boxs[:,2] * 0.5
    out[:,1] = boxs[:,1] - boxs[:,3] * 0.5
    out[:,2] = boxs[:,0] + boxs[:,2] * 0.5
    out[:,3] = boxs[:,1] + boxs[:,3] * 0.5

    return out

def bbox_iou(box1, box2, x1y1x2y2=True, GIoU=False):
    # Returns the IoU of box1 to box2. box1 is 4, box2 is nx4
    box2 = box2.t()

    # Get the coordinates of bounding boxes
    if x1y1x2y2:
        # x1, y1, x2, y2 = box1
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[0], box1[1], box1[2], box1[3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[0], box2[1], box2[2], box2[3]
    else:
        # x, y, w, h = box1
        b1_x1, b1_x2 = box1[0] - box1[2] / 2, box1[0] + box1[2] / 2
        b1_y1, b1_y2 = box1[1] - box1[3] / 2, box1[1] + box1[3] / 2
        b2_x1, b2_x2 = box2[0] - box2[2] / 2, box2[0] + box2[2] / 2
        b2_y1, b2_y2 = box2[1] - box2[3] / 2, box2[1] + box2[3] / 2

    # Intersection area
    inter_area = (torch.min(b1_x2, b2_x2) - torch.max(b1_x1, b2_x1)).clamp(0) * \
                 (torch.min(b1_y2, b2_y2) - torch.max(b1_y1, b2_y1)).clamp(0)

    # Union Area
    union_area = ((b1_x2 - b1_x1) * (b1_y2 - b1_y1) + 1e-16) + \
                 (b2_x2 - b2_x1) * (b2_y2 - b2_y1) - inter_area

    iou = inter_area / union_area  # iou
    if GIoU:  # Generalized IoU https://arxiv.org/pdf/1902.09630.pdf
        c_x1, c_x2 = torch.min(b1_x1, b2_x1), torch.max(b1_x2, b2_x2)
        c_y1, c_y2 = torch.min(b1_y1, b2_y1), torch.max(b1_y2, b2_y2)
        c_area = (c_x2 - c_x1) * (c_y2 - c_y1)  # convex area
        return iou - (c_area - union_area) / c_area  # GIoU

    return iou


if __name__ == '__main__':
    locs = getLabelFromXml('../asset/train/train_labels/mask_0000.xml')

    boxs = np.array([[1,1,5,5],[2,3,5,7]])

    boxs1 = box_xy2cwh(boxs)
    boxs2 = box_cwh2xy(boxs1)