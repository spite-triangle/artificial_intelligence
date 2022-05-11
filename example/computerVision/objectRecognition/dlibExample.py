import cv2
import dlib
import numpy as np

video = cv2.VideoCapture('./asset/face.mp4')

# 获取默认的检测
detector : dlib.fhog_object_detector = dlib.get_frontal_face_detector()


print(type(detector))

while video.isOpened():
    # 读取
    flag,frame = video.read()
    if flag == False:
        break

    frame = cv2.resize(frame,(0,0),fx=0.3,fy=0.3)
    frameGray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = detector(frame,0)

    # 标记
    for face in faces:

        y1 = face.bottom()  # detect box bottom y value
        y2 = face.top()  # top y value
        x1 = face.left()  # left x value
        x2 = face.right()  # right x value
        cv2.rectangle(frame,(x1,y1),(x2,y2),(0,0,255),2)

    cv2.imshow('face',frame)

    if cv2.waitKey(1000 // 30) == ord('q'):
        break

cv2.destroyAllWindows()



