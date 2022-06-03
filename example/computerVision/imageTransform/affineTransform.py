# %% 仿射变换
import cv2
import numpy as np

img = cv2.imread('./cat.jpeg')

img = cv2.resize(img,(0,0),fx=0.5,fy=0.5)

# warpAffine(src, M, dsize[, dst[, flags[, borderMode[, borderValue]]]]) -> dst
# M ：仿射变换矩阵
# dsize ：输出图片的大小
# flags：图片的插值算法，默认算法就不错
# borderMode：查看 图像边界扩展 小节
imgm = cv2.warpAffine(img,M=np.array([[1,0,100],[0,1,50]],dtype=np.float),dsize=(img.shape[1],img.shape[0]))
imgs = cv2.warpAffine(img,M=np.array([[0.5,0,0],[0,0.5,0]],dtype=np.float),dsize=(img.shape[1],img.shape[0]))

# center:tuple ，旋转中心
# angle，逆时针旋转角度
# scale，图片缩放值
# getRotationMatrix2D: (center: Any, angle: Any, scale: Any) -> Any
M = cv2.getRotationMatrix2D((100,100),45,1)

imgr = cv2.warpAffine(img,M=M,dsize=(img.shape[1],img.shape[0]))

cv2.imshow('affine',np.hstack((img,imgm,imgs,imgr)))

cv2.waitKey(0)
cv2.destroyAllWindows()

# %%  切变
import cv2
import numpy as np

img = cv2.imread('./cat.jpeg')

img = cv2.resize(img,(0,0),fx=0.5,fy=0.5)

M = np.array([[0.5,0.2,0],[0,0.5,0]],dtype=np.float)

imgr = cv2.warpAffine(img,M=M,dsize=(img.shape[1],img.shape[0]))

cv2.imshow('affine',imgr)

cv2.waitKey(0)
cv2.destroyAllWindows()

# %% 透视变换

import cv2
import numpy as np

img = cv2.imread('./threeBody.jpg')

srcPts = np.array([[182,207],[881,233],[1022,1337],[15,1322]],dtype=np.float32)
dstPts = np.array([[10,10],[800,10],[800,1000],[10,1000]],dtype=np.float32)
M = cv2.getPerspectiveTransform(src=srcPts,dst=dstPts)

imgr = cv2.warpPerspective(img,M,dsize=(810,1010))

imgr = cv2.resize(imgr,(0,0),fx=0.5,fy=0.5)

cv2.imshow('affine',imgr)

cv2.waitKey(0)
cv2.destroyAllWindows()
