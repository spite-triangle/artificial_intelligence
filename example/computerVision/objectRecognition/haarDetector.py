# %% haar 算法
import cv2

video = cv2.VideoCapture('./asset/face.mp4')

# 加载模型
faceDetector = cv2.CascadeClassifier('./asset/haarcascade_frontalcatface.xml')
eyeDetector = cv2.CascadeClassifer('./asset/haarcascade_eye.xml')

while video.isOpened():
    # 读取
    flag,frame = video.read()
    if flag == False:
        break

    frame = cv2.resize(frame,(0,0),fx=0.3,fy=0.3)
    frameGray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # scaleFactor：两个相邻窗口的间隔比例
    # minNeighbors：弱分类器要满足多少个，才认为是目标
    # flags：兼容旧版
    # minSize：目标对象可能的最小尺寸
    # maxSize：目标对象可能的最大尺寸
    # objects：所有目标的 x,y,w,h
    # cv.CascadeClassifier.detectMultiScale(image[, scaleFactor[, minNeighbors[, flags[, minSize[, maxSize]]]]] -> objects
    faces = faceDetector.detectMultiScale(frameGray,1.2,3,0,(200,200))
    eyes = eyeDetector.detectMultiScale(frameGray,1.1,6,0,(50,50))

    # 标记
    for (x,y,w,h) in faces:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),2)

    # 标记
    for (x,y,w,h) in eyes:
        cv2.ellipse(frame,(int(x + w/2),int(y + h/2)),(int(w/2),int(h/2)),0,0,360,(0,255,0),2)

    cv2.imshow('face',frame)

    if cv2.waitKey(1000 // 60) == ord('q'):
        break

cv2.destroyAllWindows()



