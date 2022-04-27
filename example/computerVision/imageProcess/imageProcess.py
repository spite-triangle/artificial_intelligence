# %% 金字塔
import cv2
import numpy as np

img = cv2.imread('./cat.jpeg')

imgDown = cv2.pyrDown(img)

imgUp = cv2.pyrUp(img)

print(img.shape,imgDown.shape,imgUp.shape)

# %% 轮廓
import cv2
import numpy as np
# 读取图片
img = cv2.imread('./morphology.jpg')
# 灰度图
imgGray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
# 二值化
retval,imgBinary = cv2.threshold(imgGray,127,255,cv2.THRESH_BINARY)

# 提取轮廓
binary,contours, hierarchy=cv2.findContours(imgBinary,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)

# 绘制轮廓
# drawContours(image, contours, contourIdx, color[, thickness[, lineType[, hierarchy[, maxLevel[, offset]]]]]) -> image
canva = img.copy()
cv2.drawContours(canva,contours,-1,(0,0,255),2)

cv2.imshow('contours',canva)

cv2.waitKey(0)
cv2.destroyAllWindows()

# %% 轮廓处理

# 轮廓索引
cnt = contours[0]

# 计算面积
area = cv2.contourArea(cnt)

# 计算周长
# arcLength(curve, closed) -> retval
arc = cv2.arcLength(cnt,True)

# %% 轮廓标记

canvabg = img.copy()
# 获取轮廓
cnt0 = contours[0]
# 矩形边框
startx,starty,width,height = cv2.boundingRect(cnt0)

cv2.rectangle(canvabg,(startx,starty),(startx + width,starty + height),(0,255,0),2)

# 获取轮廓
cnt1 = contours[2]
# 圆圈外框
(cx,cy),radius = cv2.minEnclosingCircle(cnt1)

cv2.circle(canvabg,(int(cx),int(cy)),int(radius),(255,0,0),2)

cv2.imshow('outline',canvabg)

cv2.waitKey(0)
cv2.destroyAllWindows()

# %% 模板匹配
import cv2
import numpy as np
img = cv2.imread('./cat.jpeg')
imgTemp = img[80:250,250:440]


cv2.imshow('template',imgTemp)

# matchTemplate(image, templ, method[, result[, mask]]) -> result
result = cv2.matchTemplate(img,imgTemp,cv2.TM_SQDIFF_NORMED)

# minMaxLoc(src[, mask]) -> minVal, maxVal, minLoc, maxLoc
minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(result)

cv2.rectangle(img,minLoc,(minLoc[ 0 ]+imgTemp.shape[ 1 ],minLoc[ 1 ]+imgTemp.shape[ 0 ]),(255,0,0),2)


cv2.imshow('match',img)

cv2.waitKey(0)
cv2.destroyAllWindows()

# %% 均衡化
import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('./cat.jpeg')

yuv =cv2.cvtColor(img,cv2.COLOR_BGR2YUV)

# 均衡化
yEqul = cv2.equalizeHist(yuv[:,:,0])

yuv[:,:,0] = yEqul

imgEual = cv2.cvtColor(yuv,cv2.COLOR_YUV2BGR)

cv2.imshow('match',np.hstack((img,imgEual)))

cv2.waitKey(0)
cv2.destroyAllWindows()

plt.figure(1,figsize=(25,10))
plt.subplot(121)
plt.hist(yuv[:,:,0].ravel(),256)
plt.subplot(122)
plt.hist(yEqul.ravel(),256)

plt.show()

# %% 自适应均衡化
import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('./cat.jpeg')

# img = cv2.bilateralFilter(img,9,sigmaColor=5,sigmaSpace=1)

yuv =cv2.cvtColor(img,cv2.COLOR_BGR2YUV)

# 生成自适应方法
# createCLAHE([, clipLimit[, tileGridSize]]) -> retval
clahe = cv2.createCLAHE(clipLimit=2.0,tileGridSize=(8,8))

yEqul = clahe.apply(yuv[:,:,0])

yuv[:,:,0] = yEqul


imgEual = cv2.cvtColor(yuv,cv2.COLOR_YUV2BGR)

cv2.imshow('match',np.hstack((img,imgEual)))

cv2.waitKey(0)
cv2.destroyAllWindows()

# %% 傅里叶变换

import cv2
import numpy as np

# 图片读取
img = cv2.imread('./cat.jpeg')
yuv =cv2.cvtColor(img,cv2.COLOR_BGR2YUV)

# 将灰度值转浮点类型
yfloat = np.float32(yuv[:,:,0])

# 傅里叶变换
# src：浮点类型数组
# flags：cv2.DFT_
# dft(src:np.float[, dst[, flags[, nonzeroRows]]]) -> dst
dft = cv2.dft(yfloat,flags=cv2.DFT_COMPLEX_OUTPUT)

# 计算模，也就是幅值
A = cv2.magnitude(dft[:,:,0],dft[:,:,1])

# 频谱中心化
shiftA = np.fft.fftshift(A)

# 幅值太大了，重新映射到 (0 - 255)，方便显示
shiftA = shiftA / shiftA.max() * 255


cv2.imshow('shit',shiftA)


cv2.waitKey(0)
cv2.destroyAllWindows()

# %% 低通滤波
import cv2
import numpy as np

# 图片读取
img = cv2.imread('./cat.jpeg')
yuv =cv2.cvtColor(img,cv2.COLOR_BGR2YUV)

# 将灰度值转浮点类型，傅里叶变换并中心化
yfloat = np.float32(yuv[:,:,0])
dft = cv2.dft(yfloat,flags=cv2.DFT_COMPLEX_OUTPUT)
dftShift = np.fft.fftshift(dft)

# 找到低频起始，中心化后频谱的中心位置
centerRow = int(dftShift.shape[0] / 2)
centerCol = int(dftShift.shape[1] / 2)

# NOTE - 高频处置为零，低频保留，然后清除对应频率幅值
mask = np.zeros(dftShift.shape,dtype=np.uint8)
mask[centerRow-50:centerRow+50,centerCol-50:centerCol+50,:] = 1
dftShift = dftShift * mask

# 反去中心。反傅里叶
dft = np.fft.ifftshift(dftShift)
idft = cv2.idft(dft)


# NOTE - 傅里叶变换结果仍然是一个复数，还要转为实数，并且还要将浮点型映射为为（0 ~ 255）之间的 uint8 类型
iyDft = cv2.magnitude(idft[:,:,0],idft[:,:,1])
iy = np.uint8(iyDft/iyDft.max() * 255)

# 还原图片,还原颜色通道
yuv[:,:,0] = iy
imgRes = cv2.cvtColor(yuv,cv2.COLOR_YUV2BGR)

cv2.imshow('low pass',np.hstack((img,imgRes)))
cv2.waitKey(0)
cv2.destroyAllWindows()


# %% 高通滤波
import cv2
import numpy as np

# 图片读取
img = cv2.imread('./cat.jpeg')
yuv =cv2.cvtColor(img,cv2.COLOR_BGR2YUV)

# 将灰度值转浮点类型，傅里叶变换并中心化
yfloat = np.float32(yuv[:,:,0])
dft = cv2.dft(yfloat,flags=cv2.DFT_COMPLEX_OUTPUT)
dftShift = np.fft.fftshift(dft)

# 找到低频起始，中心化后频谱的中心位置
centerRow = int(dftShift.shape[0] / 2)
centerCol = int(dftShift.shape[1] / 2)

# NOTE - 低频处置为零，高频保留，然后清除对应频率幅值
mask = np.ones(dftShift.shape,dtype=np.uint8)
mask[centerRow-50:centerRow+50,centerCol-50:centerCol+50,:] = 0
dftShift = dftShift * mask

# 反去中心。反傅里叶
dft = np.fft.ifftshift(dftShift)
idft = cv2.idft(dft)

# NOTE - 傅里叶变换结果仍然是一个复数，还要转为实数，并且还要将浮点型映射为为（0 ~ 255）之间的 uint8 类型
iyDft = cv2.magnitude(idft[:,:,0],idft[:,:,1])
iy = np.uint8(iyDft/iyDft.max() * 255)

# 还原图片,还原颜色通道
yuv[:,:,0] = iy
imgRes = cv2.cvtColor(yuv,cv2.COLOR_YUV2BGR)

cv2.imshow('low pass',np.hstack((img,imgRes)))
cv2.waitKey(0)
cv2.destroyAllWindows()