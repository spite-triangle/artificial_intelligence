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
