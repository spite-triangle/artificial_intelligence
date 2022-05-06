# %% 差帧法
import cv2
import numpy as np

bgimg = cv2.imread('./asset/diffFrame_background.jpg')
fgimg = cv2.imread('./asset/diffFrame_people.jpg')

bgimgGray = cv2.cvtColor(bgimg,cv2.COLOR_BGR2GRAY)
fgimgGray = cv2.cvtColor(fgimg,cv2.COLOR_BGR2GRAY)

diff = np.abs(cv2.subtract(bgimgGray,fgimgGray))

mask = np.zeros_like(diff)
mask[diff > 20] = 255

mask = cv2.cvtColor(mask,cv2.COLOR_GRAY2BGR)

cv2.imshow('mask',mask)
cv2.waitKey(0)
cv2.destroyAllWindows()

# %% 高斯混合模型
import cv2

video = cv2.VideoCapture('./asset/test.mp4')

# 高斯混合模型
GaussModel = cv2.createBackgroundSubtractorMOG2()

# 形态学卷积核
kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))

# 视频保存格式
videoForm = cv2.VideoWriter_fourcc(*'mp4v')
# 保存视频的类，输入参数为：
# 保存路径，保存格式，保存的视频帧数，（宽度像素，高度像素） 
videoSave = cv2.VideoWriter('./asset/out.mp4',videoForm,30,(960, 540))

while video.isOpened():

    # 读取视频
    flag,frame = video.read()
    if flag == False:
        break

    frame = cv2.resize(frame,(0,0),fx=0.5,fy=0.5)

    print(frame.shape)

    # 高斯混合
    fgMask = GaussModel.apply(frame)

    # 形态学处理
    fgMask = cv2.morphologyEx(fgMask,cv2.MORPH_OPEN,kernel,iterations=1)
    fgMask = cv2.morphologyEx(fgMask,cv2.MORPH_CLOSE,kernel,iterations=3)

    # 动态区域标记
    frame[fgMask == 255] = [255,255,0]

    # 保存图片
    videoSave.write(frame)

    cv2.imshow('video',frame)
    if cv2.waitKey(1000 // 30) == ord('q'):
        break

videoSave.release()
video.release()
cv2.destroyAllWindows()