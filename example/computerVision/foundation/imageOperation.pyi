# %% 导入库
from turtle import pd
import cv2
import numpy as np
import matplotlib.pyplot as plt

# %% 图像的输入与输出

def showImage(name:str,image):
    cv2.imshow(name,image)
    cv2.waitKey(0) 
    cv2.destroyAllWindows()

# 读取图
img = cv2.imread('./asset/cat.jpeg',cv2.IMREAD_COLOR)

# cv2 显示图片
showImage('cat',img)

# 保存图
# cv2.imwrite('./asset/cv2_cat.jpeg')

# %% 图片数据格式

print(img)

# 图片数据格式
print(type(img))

# 数组的维度
print(img.shape)
print(img.dtype)


# %% 数据操作

# 图片切割
showImage('cut_cat',img[ 100:200,:,: ])

# 通道拆分
b,g,r = cv2.split(img)
print(b)
print(img[:,:,0])

# 单通道图片显示
b = img[:,:,0]
g = img[:,:,1] * 0
r = img[:,:,2] * 0
imgB = cv2.merge((b,g,r)) 
showImage('b',imgB)

# %% 加和

img = img + imgB

cv2.addWeighted(np.array([254],dtype=np.uint8),0.1,np.array([255],dtype=np.uint8),1,0)

# %% 像素尺寸

imgResize = cv2.resize(img,(0,0),fx=1.5,fy=1)
showImage('resize',imgResize)


# %% 阈值操作

ret,imgBinary = cv2.threshold(img,100,220,cv2.THRESH_TRUNC)
showImage('THRESH_BINARY',imgBinary)

# %% 边界填充
extendImg = cv2.copyMakeBorder(img,50,50,50,50,cv2.BORDER_REFLECT)
showImage('extend',extendImg)


