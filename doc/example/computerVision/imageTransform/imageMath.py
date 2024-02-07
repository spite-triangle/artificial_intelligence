# %% 非运算

import cv2
import numpy as np

img = cv2.imread('./cat.jpeg')

# 非运算
imgR = cv2.bitwise_not(img)

cv2.imshow('not',np.hstack((img,imgR)))

cv2.waitKey(0)
cv2.destroyAllWindows()

# %% 翻转

import cv2
import numpy as np

img = cv2.imread('./cat.jpeg')

img = cv2.resize(img,(0,0),fx=0.5,fy=0.5)

# 翻转
# flip(src, flipCode[, dst]) -> dst
# flipCode > 0：
# flipCode < 0：
# flipCode = 0：
img0 = cv2.flip(img,0)
imgLow0 = cv2.flip(img,-1)
imgGreat0 = cv2.flip(img,1)

cv2.imshow('flip',np.hstack((img,img0,imgLow0,imgGreat0)))

cv2.waitKey(0)
cv2.destroyAllWindows()

# %% 旋转
import cv2
import numpy as np

img = cv2.imread('./cat.jpeg')

# 旋转
# rotate(src, rotateCode[, dst]) -> dst
# roteCode：cv2.ROTATE_
imgr = cv2.rotate(img,cv2.ROTATE_180)

cv2.imshow('flip',imgr)

cv2.waitKey(0)
cv2.destroyAllWindows()

# %% 图片坐标

import matplotlib.pyplot as plt

img = plt.imread('./cat.jpeg')


ax = plt.gca() 
ax.xaxis.set_ticks_position('top')  #将x轴的位置设置在顶部
ax.invert_xaxis()  #x轴反向
ax.yaxis.set_ticks_position('left') # 将y轴的位置设置在右边
ax.invert_yaxis()  #y轴反向

plt.title('with pixel')
plt.ylabel('height pixel')

plt.imshow(img)

