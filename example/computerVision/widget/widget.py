# %% 创建 trackbar

import cv2
from matplotlib import image

# trackbar 改变时的回调函数
def onTrackbarChange(value):
    print(value)

# 创建界面
cv2.namedWindow('trackbar',cv2.WINDOW_NORMAL)
cv2.resizeWindow('trackbar',width=640,height=360)

# 创建trackbar
cv2.createTrackbar('bar','trackbar',0,255,onTrackbarChange)

# 读取trackbar 的值
value = cv2.getTrackbarPos('bar','trackbar')
print(value)

cv2.waitKey(0)
cv2.destroyAllWindows()
# %% 画直线、圆形、矩形

import cv2
import numpy as np

# 创建窗口
cv2.namedWindow('draw_shape',cv2.WINDOW_NORMAL)
cv2.resizeWindow('draw_shape ',width=640,height=360)

# 必须先有一张背景图，用来当画布
canvas = np.zeros(shape=(360,640,3),dtype=np.uint8)
canvas[:,:] = [255,255,0]

# 矩形
# rectangle(canvas:img, pt1:tuple, pt2:tuple, color[, thickness[, lineType[, shift]]]) -> img
imgRect = cv2.rectangle(canvas,(20,40),(100,100),(255,0,0),3)

# 圆形
# circle(canvas:img, center:tuple:, radius, color[, thickness[, lineType[, shift]]]) -> img
imgCircle = cv2.circle(canvas,(200,60),50,(0,0,255),3,16)

# 直线
# 
# img = cv2.line()
# line(canvas:img, pt1:tuple, pt2:tuple, color[, thickness[, lineType[, shift]]]) -> img
imgLine2 = cv2.line(canvas,(20,150),(200,170),(255,0,0),20,2)
imgLine8 = cv2.line(canvas,(20,200),(200,220),(255,0,0),20,8)
imgLine16 = cv2.line(canvas,(20,250),(200,270),(255,0,0),20,16)

print(canvas.__array_interface__['data'])
print(imgRect.__array_interface__['data'])
print(imgCircle.__array_interface__['data'])
print(imgLine2.__array_interface__['data'])

cv2.imshow('draw_shape',canvas)
cv2.waitKey(0)
cv2.destroyAllWindows()

# %% 椭圆
import cv2
import numpy as np

# 创建窗口
cv2.namedWindow('draw_shape',cv2.WINDOW_NORMAL)
cv2.resizeWindow('draw_shape ',width=640,height=360)

# 必须先有一张背景图，用来当画布
canvas = np.zeros(shape=(360,640,3),dtype=np.uint8)
canvas[:,:] = [255,255,0]

# 椭圆
# ellipse(img, center:tuple, axes:tuple, angle, ArcStartAngle, ArcEndAngle, color[, thickness[, lineType[, shift]]]) -> img
cv2.ellipse(canvas,(100,100),(60,80),90,0,270,(255,0,0))

cv2.imshow('draw_shape',canvas)
cv2.waitKey(0)
cv2.destroyAllWindows()

# %% 椭圆
import cv2
import numpy as np

# 创建窗口
cv2.namedWindow('draw_shape',cv2.WINDOW_NORMAL)
cv2.resizeWindow('draw_shape ',width=640,height=360)

# 必须先有一张背景图，用来当画布
canvas = np.zeros(shape=(360,640,3),dtype=np.uint8)
canvas[:,:] = [255,255,0]

# 多边形
# polylines(img, [pts:numpy.ndarray], isClosed, color[, thickness[, lineType[, shift]]]) -> img
pts1 = np.array([ (20,60),(300,150),(50,300) ])
pts2 = np.array([ (400,60),(300,100) ])
cv2.polylines(canvas,[pts1,pts2],True,(255,0,0))

cv2.imshow('draw_shape',canvas)
cv2.waitKey(0)
cv2.destroyAllWindows()

# %% 英文文本

import cv2
import numpy as np

# 创建窗口
cv2.namedWindow('draw_shape',cv2.WINDOW_NORMAL)
cv2.resizeWindow('draw_shape ',width=640,height=360)

# 必须先有一张背景图，用来当画布
canvas = np.zeros(shape=(360,640,3),dtype=np.uint8)
canvas[:,:] = [255,255,0]

# 放置文本
# putText(img, text, pos:tuple, fontFace, fontScale, 
#         color[, thickness[, lineType[, bottomLeftOrigin]]]) -> img
cv2.putText(canvas,'hellow world',(100,100),cv2.FONT_HERSHEY_COMPLEX,1,(255,0,0))

cv2.imshow('draw_shape',canvas)
cv2.waitKey(0)
cv2.destroyAllWindows()


# %% 中文文本
import cv2
import numpy as np
from PIL import Image,ImageFont,ImageDraw

# 创建窗口
cv2.namedWindow('draw_shape',cv2.WINDOW_NORMAL)
cv2.resizeWindow('draw_shape ',width=640,height=360)

# 必须先有一张背景图，用来当画布
canvas = np.zeros(shape=(360,640,3),dtype=np.uint8)
canvas[:,:] = [255,255,0]

# 导入字体
font = ImageFont.truetype('./asset/eva_font.otf',size=35)
# 创建画布
canvasBg = Image.fromarray(canvas)
# 创建画笔
brush = ImageDraw.Draw(canvasBg)
# 写入中文
brush.text(xy=(100,100),text="使徒襲来",font=font,fill=(255,0,0,0))

# 转换回cv2矩阵
canvas = np.array(canvasBg)

cv2.imshow('draw_shape',canvas)
cv2.waitKey(0)
cv2.destroyAllWindows()


