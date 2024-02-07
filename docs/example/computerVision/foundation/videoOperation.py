# %%
from pickletools import uint8
import cv2
from cv2 import VideoWriter

# %% 读取视频

# 读取视频
video = cv2.VideoCapture('./asset/sarsa_clip.mp4')

# 视频读取
while video.isOpened():
    # 读取一帧
    flag,frame = video.read()

    # 显示
    if flag == True:
        cv2.imshow('video',frame)

    # 控制播放速度：以 60 帧的速度进行图片显示
    if cv2.waitKey(1000 // 60) == ord('q'):
        break 

# 释放
video.release()

cv2.destroyAllWindows()

# %% 摄像头

# 读取视频
video = cv2.VideoCapture(0)

# 视频读取
while video.isOpened():

    # 读取一帧
    flag,frame = video.read()

    # 是否读取成功
    if flag == True:
        # 显示
        cv2.imshow('video',frame)

    # 控制播放速度：以 60 帧的速度进行图片显示
    if cv2.waitKey(1000 // 60) == ord('q'):
        break 

# 释放
video.release()

cv2.destroyAllWindows()
# %% 视频保存

# 读取视频
video = cv2.VideoCapture(0)

# 视频保存格式
videoForm = cv2.VideoWriter_fourcc(*'mp4v')

# 保存视频的类，输入参数为：
# 保存路径，保存格式，保存的视频帧数，（宽度像素，高度像素） 
videoSave = cv2.VideoWriter('./asset/capture.mp4',videoForm,24,(640, 480))

# 视频读取
while video.isOpened():

    # 读取一帧
    flag,frame = video.read()

    # 是否读取成功
    if flag == True:
        # 显示
        cv2.imshow('video',frame)
        # 保存
        videoSave.write(frame)

    # 控制播放速度：以 60 帧的速度进行图片显示
    if cv2.waitKey(1000 // 60) == ord('q'):
        break 

# 释放
videoSave.release()
video.release()

cv2.destroyAllWindows()

# %% 鼠标事件
import cv2
import numpy as np
# 定义回调函数
def mouse_callback(event,x,y,flags,userdata:any):
    if event == cv2.EVENT_LBUTTONDOWN:
        print(event,x,y,flags,userdata)


# 创建窗口
cv2.namedWindow('Event Test',cv2.WINDOW_NORMAL)
cv2.resizeWindow('Event Test',width=640,height=380)

# 鼠标事件指定回调函数
cv2.setMouseCallback('Event Test',mouse_callback,"userdata")

# 生成一个背景图片
bg = np.zeros(shape=(380,640,3),dtype=np.uint8)

while True:
    cv2.imshow('Event Test',bg)
    if cv2.waitKey(0) == ord('q'):
        break

cv2.destroyAllWindows()

# %%
