import cv2
import numpy as np

video = cv2.VideoCapture('./asset/opticalFlow.mp4')


# 视频保存格式
videoForm = cv2.VideoWriter_fourcc(*'mp4v')
# 保存视频的类，输入参数为：
# 保存路径，保存格式，保存的视频帧数，（宽度像素，高度像素） 
videoSave = cv2.VideoWriter('./asset/out.mp4',videoForm,30,(538, 822))

# 预读一帧
flag,lastFrame = video.read()
lastFrame = cv2.resize(lastFrame,(0,0),fx=0.5,fy=0.5)
lastFrameGray = cv2.cvtColor(lastFrame, cv2.COLOR_BGR2GRAY)

print(lastFrame.shape)

# 获取角点
# qualityLevel：筛选角点的阈值，评估 lambda1 与 lambda2
# minDistance：在这个距离内，有比当前角点更好的，那就不要当前这个点了
# goodFeaturesToTrack(image, maxCorners, qualityLevel, minDistance[, 
#                       corners[, mask[, blockSize[, useHarrisDetector[, k]]]]]) -> corners
lastPoints = cv2.goodFeaturesToTrack(lastFrameGray,100,qualityLevel=0.2,minDistance=7)

# 路径
track = np.zeros_like(lastFrame)

count = 0
while video.isOpened():

    # 读取视频
    flag,curFrame = video.read()
    if flag == False:
        break
    
    # 重新找角点
    if count % 15 == 0:
        lastPoints = cv2.goodFeaturesToTrack(lastFrameGray,50,qualityLevel=0.1,minDistance=30)
        track[:,:,:] = 0

    curFrame = cv2.resize(curFrame,(0,0),fx=0.5,fy=0.5)
    curFrameGray = cv2.cvtColor(curFrame,cv2.COLOR_BGR2GRAY)

    # 光流估计
    # winSize：领域区间
    # maxLevel：金字塔层数
    # curPoints：上一步特征点当前跑到哪个位置了
    # state：上一步的特征点当前找没找到
    curPoints,statas,err = cv2.calcOpticalFlowPyrLK(lastFrameGray,curFrameGray,lastPoints,None,winSize=(15,15),maxLevel=2)

    # 画路径
    trackLastPts = lastPoints[statas==1]
    trackCurPts = curPoints[statas==1]
    for i in range(len(trackCurPts)):
        lastPt = trackLastPts[i].ravel()
        curPt = trackCurPts[i].ravel()
        # 画线
        cv2.line(track,(int(lastPt[0]),int(lastPt[1])),(int(curPt[0]),int(curPt[1])),(255,20,0),3,8)
        # 标记关键点
        cv2.circle(curFrame,(int(curPt[0]),int(curPt[1])),5,(0,0,255),1)
        curFrame = cv2.add(track,curFrame)

    # 保存图片
    videoSave.write(curFrame)

    cv2.imshow('video',curFrame)
    if cv2.waitKey(1000 // 30) == ord('q'):
        break

    # 刷新
    lastPoints = curPoints
    lastFrameGray = curFrameGray
    count = count + 1

videoSave.release()
video.release()
cv2.destroyAllWindows()