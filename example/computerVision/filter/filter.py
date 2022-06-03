# %% 方盒滤波
import cv2 
import numpy as np

img = cv2.imread('./cat.jpeg')
img = cv2.resize(img,(0,0),fx=0.6,fy=0.6)

# cv2.boxFilter(src, ddepth, ksize:tuple[, dst[, anchor[, normalize[, borderType]]]]) -> dst
imgbox = cv2.boxFilter(img,-1,(5,5),normalize=True)


cv2.imshow('boxfilter',np.hstack((img,imgbox)))
cv2.waitKey(0)
cv2.destroyAllWindows()

# %% 高斯滤波
import cv2 
import numpy as np

img = cv2.imread('./suzanne.png')
# img = cv2.resize(img,(0,0),fx=0.6,fy=0.6)

# GaussianBlur(src, ksize, sigmaX[, dst[, sigmaY[, borderType]]]) -> dst
imgGaus = cv2.GaussianBlur(img,(5,5),sigmaX=1)


cv2.imshow('Gaussfilter',np.hstack((img,imgGaus)))
cv2.waitKey(0)
cv2.destroyAllWindows()


# %% 中值滤波

import cv2 
import numpy as np

img = cv2.imread('./suzanne.png')
# img = cv2.resize(img,(0,0),fx=0.6,fy=0.6)

# medianBlur(src, ksize[, dst]) -> dst
imgm = cv2.medianBlur(img,7)

cv2.imshow('Gaussfilter',np.hstack((img,imgm)))
cv2.waitKey(0)
cv2.destroyAllWindows()

# %% 双边滤波

import cv2 
import numpy as np

img = cv2.imread('./gaussDisturbance.jpg')
img = cv2.resize(img,(0,0),fx=0.6,fy=0.6)

sigma = 250
ks = 13
# bilateralFilter(src, d, sigmaColor, sigmaSpace[, dst[, borderType]]) -> dst
imgb = cv2.bilateralFilter(img,ks,sigmaColor=sigma,sigmaSpace=8)
imgg = cv2.GaussianBlur(img,(ks,ks),sigmaX=sigma)

cv2.imshow('bilateralFilter',np.hstack((img,imgb,imgg)))
cv2.waitKey(0)
cv2.destroyAllWindows()


# %%  sobel 算子
import cv2 
import numpy as np

img = cv2.imread('./cat.jpeg',cv2.IMREAD_GRAYSCALE)
img = cv2.resize(img,(0,0),fx=0.6,fy=0.6)

# Sobel(src, ddepth, dx, dy[, dst[, ksize:int[, scale[, delta[, borderType]]]]]) -> dst
# 竖着的边界
imgv = cv2.Sobel(img,cv2.CV_16S,dx=1,dy=0,ksize=3)
imgv = cv2.convertScaleAbs(imgv)

# 横着的边界
imgh = cv2.Sobel(img,cv2.CV_16S,dx=0,dy=1,ksize=3)
imgh = cv2.convertScaleAbs(imgh)

# 边界叠加
imga = cv2.add(imgh,imgv)

cv2.imshow('sobel',np.hstack((img,imgv,imgh,imga)))
cv2.waitKey(0)
cv2.destroyAllWindows()

# %%  拉普拉斯算子
import cv2 
import numpy as np

# img = np.zeros((200,200,3),dtype=np.uint8)
# img[50:150,50:150] = 255

img = cv2.imread('./gray.jpg',cv2.IMREAD_GRAYSCALE)

# img = cv2.resize(img,(0,0),fx=0.6,fy=0.6)

imgr = cv2.Laplacian(img,ddepth=cv2.CV_32F,ksize=5)
imgr = cv2.convertScaleAbs(imgr)

imgh = cv2.Sobel(img,cv2.CV_32F,dx=1,dy=0,ksize=5)
imgh = cv2.convertScaleAbs(imgh)

cv2.imshow('laplace',np.hstack(( img,imgr ,imgh)))
cv2.waitKey(0)
cv2.destroyAllWindows()

# %%  canny
import cv2 
import numpy as np

# img = np.zeros((200,200,3),dtype=np.uint8)
# img[50:150,50:150] = 255

img = cv2.imread('./gaussDisturbance.jpg',cv2.IMREAD_GRAYSCALE)

# img = cv2.resize(img,(0,0),fx=0.6,fy=0.6)

img= cv2.bilateralFilter(img,7,sigmaColor=10000,sigmaSpace=1.6)

imgr = cv2.Canny(img,40,120)

cv2.imshow('sobel',np.hstack(( img,imgr )))
cv2.waitKey(0)
cv2.destroyAllWindows()