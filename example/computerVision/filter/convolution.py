# %% 边缘监测

import cv2 
import numpy as np

img = cv2.imread('./cat.jpeg',cv2.IMREAD_GRAYSCALE)

img = cv2.resize(img,(0,0),fx=0.5,fy=0.5)

# filter2D(src, ddepth, kernel[, dst[, anchor[, delta[, borderType]]]]) -> dst
# 竖向监测
kv = np.array([[1,0,-1],[1,0,-1],[1,0,-1]],dtype=np.float)
imgv = cv2.filter2D(img,cv2.CV_32F,kv)
imgv = cv2.convertScaleAbs(imgv)

# 横向监测
kh = np.array([[1,1,1],[0,0,0],[-1,-1,-1]],dtype=np.float)
imgh = cv2.filter2D(img,cv2.CV_32F,kh)
imgh = cv2.convertScaleAbs(imgh)

cv2.imshow('convolution',np.hstack(( img,imgv,imgh )))

cv2.waitKey(0)
cv2.destroyAllWindows()

# %% 边缘监测

import cv2 
import numpy as np

img = cv2.imread('./cat.jpeg',cv2.IMREAD_GRAYSCALE)

img = cv2.resize(img,(0,0),fx=0.5,fy=0.5)

# filter2D(src, ddepth, kernel[, dst[, anchor[, delta[, borderType]]]]) -> dst
# 竖向监测
kv = np.array([[0,0,0],[1,-2,1],[0,0,0]],dtype=np.float)
imgv = cv2.filter2D(img,cv2.CV_16S,kv)
imgv = cv2.convertScaleAbs(imgv)

# 横向监测
kh = np.array([[0,1,0],[0,-2,0],[0,1,0]],dtype=np.float)
imgh = cv2.filter2D(img,cv2.CV_16S,kh)
imgh = cv2.convertScaleAbs(imgh)

imga = cv2.add(imgv,imgh)

cv2.imshow('convolution',np.hstack(( img,imgv,imgh,imga *2 )))

cv2.waitKey(0)
cv2.destroyAllWindows()
