# %% 边缘监测

import cv2 
import numpy as np

img = cv2.imread('./cat.jpeg',cv2.IMREAD_GRAYSCALE)

img = cv2.resize(img,(0,0),fx=0.5,fy=0.5)

# filter2D(src, ddepth, kernel[, dst[, anchor[, delta[, borderType]]]]) -> dst
# 竖向监测
kv = np.array([[1,0,-1],[1,0,-1],[1,0,-1]],dtype=np.float)
imgv = cv2.filter2D(img,-1,kv)
# 横向监测
kh = np.array([[1,1,1],[0,0,0],[-1,-1,-1]],dtype=np.float)
imgh = cv2.filter2D(img,-1,kh)

cv2.imshow('convolution',np.hstack(( img,imgv,imgh )))

cv2.waitKey(0)
cv2.destroyAllWindows()

