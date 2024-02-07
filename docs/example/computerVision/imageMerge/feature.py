# %% SIFT 关键点查找
import cv2
import utils
import numpy as np

img = cv2.imread('./cat.jpeg')
imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 创建 SIFT 算法
# SIFT_create([, nfeatures[, nOctaveLayers[, contrastThreshold[, edgeThreshold[, sigma]]]]]) -> retval
sift = cv2.SIFT_create()

# 查找关键点位置
kp = sift.detect(imgGray,None)

# 计算特征
# compute(img,KeyPoints:tuple) -> KeyPoints:tuple, descriptors:np.ndarray
kp,des = sift.compute(imgGray,kp)

# 绘制关键点
# drawKeypoints(image, keypoints, outImage[, color[, flags]]) -> outImage
cv2.drawKeypoints(img,kp,img)

utils.imshow(img)

# %% BF关键点匹配

import cv2
import utils
import numpy as np

img1 = cv2.imread('./asset/06.jpg')
img2 = cv2.imread('./asset/6.jpg')
scale = 0.7
img1 = cv2.resize(img1,(0,0),fx=scale,fy=scale)
img2 = cv2.resize(img2,(0,0),fx=scale,fy=scale)
img1Gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
img2Gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

sift = cv2.SIFT_create()
kp1,des1 = sift.detectAndCompute(img1Gray,None)
kp2,des2 = sift.detectAndCompute(img2Gray,None)

# 建立匹配算法
bf = cv2.BFMatcher(crossCheck=True)

# 匹配
matchRes = bf.match(des1,des2)
matchRes = sorted(matchRes,key=lambda x:x.distance)

print(type(matchRes))

# 绘制匹配
# drawMatches(img1, keypoints1, img2, keypoints2, matches1to2, 
#           outImg[, matchColor[, singlePointColor[, matchesMask[, flags]]]]) -> outImg
imgMatch = cv2.drawMatches(img1,kp1,img2,kp2,matchRes[:1],None,flags=2)

utils.imshow(imgMatch)

# %% 1对多匹配
import cv2
import utils
import numpy as np

img1 = cv2.imread('./asset/6.jpg')
img2 = cv2.imread('./asset/06.jpg')
scale = 0.7
img1 = cv2.resize(img1,(0,0),fx=scale,fy=scale)
img2 = cv2.resize(img2,(0,0),fx=scale,fy=scale)
img1Gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
img2Gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

sift = cv2.SIFT_create()
kp1,des1 = sift.detectAndCompute(img1Gray,None)
kp2,des2 = sift.detectAndCompute(img2Gray,None)

# 建立匹配算法
bf = cv2.BFMatcher()

# 匹配
# k：一张图的关键点可以与另一张图的 k 个关键点相匹配
# knnMatch(imgSrc,imgTemp,k) -> res
matchRes = bf.knnMatch(des1,des2,k=2)

print(type(matchRes))

goodMatchs=[]
for m, n in matchRes:
    if m.distance < 0.75*n.distance:
        goodMatchs.append([m]) 

# 绘制匹配
# drawMatchesKnn(img1, keypoints1, img2, keypoints2, matches1to2, 
#           outImg[, matchColor[, singlePointColor[, matchesMask[, flags]]]]) -> outImg
imgMatch = cv2.drawMatchesKnn(img1,kp1,img2,kp2,goodMatchs,None,flags=2)

utils.imshow(imgMatch)