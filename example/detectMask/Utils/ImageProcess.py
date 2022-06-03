import os
import sys
workPath = os.path.dirname(os.path.dirname(__file__))
if workPath not in sys.path:
    sys.path.append(workPath)

from PIL import Image
import torch
import torchvision
import cv2
import numpy as np
from config import INPUT_SIZE
import Utils.BoxProcess as BoxProcess
import config

# torch 中的图片转换工具
toTensor = torchvision.transforms.ToTensor()
resize = torchvision.transforms.Resize(INPUT_SIZE)
toPIL = torchvision.transforms.ToPILImage()


def drawBoundingBoxsAndClasses(image,objs:torch.Tensor,padding=[0,0],scale:float=1.0) :
    """再图片上绘制出 bounding box 和 物体类型

    Args:
        image : 要添加 bounding box 和 物体类别的图片
        objs (torch.Tensor): bounding box 信息与物体信息
        padding : 图片统一化，需要添加的像素，[上下，左右]
        scale (float): 缩放值

    Returns:
        np.ndarray : 添加了标注框的图片
    """    
    # bounding boxs [[ x1,y1,x2,y2,classScore,classIndex ]]
    boxs = (objs[:,0:4] / scale - torch.tensor([ [ padding[1],padding[0],padding[1],padding[0] ] ], device=config.RUN_DEVICE))

    # 类别名称
    classNames = []
    for index in objs[:,5]:
        classNames.append(config.NAME_CLASSFICATIONS[int(index)])

    # 绘制矩形框
    return drawBoundingBoxsAndLabel(toCV(image),boxs,classNames)

def drawText(image,pos,text,fontFace=cv2.FONT_HERSHEY_SIMPLEX,fontScale=0.7,thickness=2,
                            frontground=(217, 85, 167),background=(90, 235,93))-> np.ndarray:
    """添加文本框

    Args:
        image (np.ndarray): 图片
        pos (tuple): 文字左下角坐标
        text (str): 文本
        fontFace (cv2.FONT_, optional): 字体. Defaults to cv2.FONT_HERSHEY_SIMPLEX.
        fontScale (float, optional): _description_. Defaults to 0.6.
        thickness (int, optional): 字体厚度. Defaults to 1.
        frontground (tuple, optional): 前景色. Defaults to (217, 85, 167).
        background (tuple, optional): 背景色. Defaults to (90, 235,93).

    Returns:
        img (np.ndarray) : 图片 
    """    
    # 计算文本的宽高，baseLine
    retval, baseLine = cv2.getTextSize(text,fontFace=fontFace,fontScale=fontScale, thickness=thickness)
    # 计算覆盖文本的矩形框坐标
    topleft = (pos[0], pos[1] - retval[1] - 5)
    bottomright = (topleft[0] + retval[0], topleft[1] + retval[1] + 5)
    cv2.rectangle(image, (topleft[0], topleft[1] - baseLine), bottomright,thickness=-1, color=background)
    # 绘制文本
    return cv2.putText(image, text, (pos[0], pos[1]-baseLine), fontScale=fontScale,fontFace=fontFace, thickness=thickness, color=frontground)


def drawBoundingBoxsAndLabel(image, boxs,labels:str=None, frontground=(217, 85, 167),background=(90, 235,93)) -> np.ndarray:
    """在图片上绘制 bounding box

    Args:
        image (np.array): 图片
        boxs : 外接矩形，[ [x1,y1,x2,y2] ]
        labels (str)：外接矩形标签
        frontground (tuple, optional): 前景色. Defaults to (217, 85, 167).
        background (tuple, optional): 背景色. Defaults to (90, 235,93).
    Returns:
        np.ndarray: 
    """
    for i,box in enumerate(boxs):
        # 带有标签就绘制
        if labels is not None:
            image = drawText(image,(int(box[0].item()), int(box[1].item())),labels[i],background=background,frontground=frontground)

        cv2.rectangle(image, (int(box[0].item()), int(box[1].item())), (int(box[2].item()), int(box[3].item())), background, 2)

    return image


def toCV(image) -> np.ndarray:
    """将 PIL image、tensor 图片转为 OpenCV 图片格式

    Args:
        image (PIL image, tensor): 图片

    Returns:
        np.ndarray: opencv 的图片，通道为 BGR
    """
    if isinstance(image,np.ndarray):
        return image

    # 是否是tensor
    if isinstance(image, torch.Tensor):
        image = toPIL(image).convert('RGB')

    # 是否是 PIL image
    if isinstance(image, Image.Image):
        image = np.array(image, dtype=np.uint8)


    # 颜色通道转换
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image


def showCV(image, title='untitled'):
    """利用OpenCV显示图片

    Args:
        image (np.ndarray): 图片
        title (str, optional): 图片标题. Defaults to 'untitled'.
    """
    # 显示图片
    cv2.imshow(title, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def normlizeSingleImage(image, paddingValue: float = 0.5) -> tuple:
    """将输入图片归一化：添加 padding 、缩放尺寸、转 tensor

    图片添加 padding 的思路：
    - 比较原始图片按照宽高大小；
    - 以大的一方为原始图缩放为 INPUT_SIZE 的基准；
    - 小的一方根据 INPUT_SIZE 计算要填充多少 padding 才能使得原始图片比列与 INPUT_SIZE 比列一样；
    - 原始图添加 padding ，利用 transform 缩放为 INPUT_SIZE

    Args:
        image (np.ndarray,PIL image): 输入图片
        paddingValue (float, optional): padding 的填充值. Defaults to 0.5.

    Returns:
        tensor: 归一化后的tensor图片
        padding:  [ 上下(y)。左右(x) ] padding 填充位数
        scale: INPUT_SIZE / 原始图片尺寸
    """
    # 图片转为 Tensor
    img = toTensor(image)
    # resize高
    _, height,width = img.shape

    # 上下，左右
    padding = [0, 0]
    if width >= height:
        # 按照输入图片的宽度进行缩放
        rate = 1.0 * INPUT_SIZE[0] / width

        # 将 INPUT_SIZE 放大，计算原始图上下位置加 padding
        padding[0] = int((INPUT_SIZE[1] / rate - height)/2)

        # 生成原图添加了padding的画布
        outImg = torch.ones(
            (3,height+2*padding[0],width), dtype=torch.float32) * paddingValue

        # 将原图塞进去
        outImg[:,padding[0]:height+padding[0],:] = img
    elif width < height:
        # 按照输入图片的高度进行缩放
        rate = 1.0 * INPUT_SIZE[1] / height

        # 将 INPUT_SIZE 放大，计算原始图在左右位置加 padding
        padding[1] = int((INPUT_SIZE[0] / rate - width)/2)

        # 生成原图添加了padding的画布
        outImg = torch.ones(
            (3, height, width+2*padding[1]), dtype=torch.float32) * paddingValue

        # 将原图塞进去
        outImg[:, :,padding[1]:width+padding[1]] = img

    # 计算真正的缩放比列，上面计算的比列可能会存在一两个像素的误差
    scaleRate = 1.0 * INPUT_SIZE[0] / outImg.shape[1]

    # 将添加了 padding 的原图缩放为 INPUT_SIZE 大小
    outImg = resize(outImg)

    return outImg, np.array(padding), scaleRate


if __name__ == '__main__':
    img = Image.open('./asset/train/train_images/mask_0000.jpg')

    boxs = torch.tensor([[173, 168, 289, 309],[653, 195, 749, 301]],dtype=torch.int)

    # 在原图上画矩形框
    showCV(drawBoundingBoxsAndLabel(toCV(img), boxs,labels=['test','test']))

    # 归一化输入
    out, padding, scale = normlizeSingleImage(img)
    padBoxs = BoxProcess.addPaddingToBoxs(boxs, padding, scale)

    # 在归一化后的图片上绘制矩形框
    showCV(drawBoundingBoxsAndLabel(toCV(out), padBoxs))
    print(padBoxs)
