import cv2
import dlib
import numpy as np

video = cv2.VideoCapture('./asset/face.mp4')

# 视频保存格式
videoForm = cv2.VideoWriter_fourcc(*'mp4v')
# 保存视频的类，输入参数为：
# 保存路径，保存格式，保存的视频帧数，（宽度像素，高度像素） 
videoSave = cv2.VideoWriter('./asset/capture.mp4',videoForm,30,(576, 324))

# 获取默认的检测
detector = dlib.get_frontal_face_detector()

# 追踪器
tracker = dlib.correlation_tracker()

# 读取第一帧
flag,frame = video.read()
frame = cv2.resize(frame,(0,0),fx=0.3,fy=0.3)
frameGray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

# 定位人脸
face = detector(frame,0)[0]

#  启动追踪
tracker.start_track(frame,face)

while video.isOpened():
    # 读取
    flag,frame = video.read()
    if flag == False:
        break

    frame = cv2.resize(frame,(0,0),fx=0.3,fy=0.3)
    frameGray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 更新追踪
    # faces = detector(frameGray,0)
    tracker.update(frame)
    face:dlib.drectangle = tracker.get_position()

    # 标记
    y1 = face.bottom()  # detect box bottom y value
    y2 = face.top()  # top y value
    x1 = face.left()  # left x value
    x2 = face.right()  # right x value
    cv2.rectangle(frame,(int(x1),int( y1 )),(int( x2 ),int( y2 )),(0,0,255),2)

    cv2.imshow('face',frame)
    videoSave.write(frame)
    if cv2.waitKey(1000//30) == ord('q'):
        break

videoSave.release()
cv2.destroyAllWindows()



