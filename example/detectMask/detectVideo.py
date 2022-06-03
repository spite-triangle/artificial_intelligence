import cv2
import torch
import numpy as np
from PIL import Image
import Utils.BoxProcess as BoxProcess
import Utils.ImageProcess as ImageProcess
import Utils.PostProcess as PostProcess
import Model.Network as Network
import config
import time

def detectObjsFromFrame(model:Network.Darknet53,frame:np.ndarray):
    """从一张图片中检测出目标物体

    Args:
        model (Network.Darknet53): yolo v3 模型
        images (np.ndarray): 从视频中读取的一帧图片 ( h,w,[B,G,R] )

    Returns:
        objs : 目标检测结果 [ x1,y1,x2,y2,classScore,classIndex ]
        padding ：填充的像素
        scale ：图片缩放
        img: 调整后的图片
    """    
    # 转换图片格式 np.ndarray 转 PIL
    img = Image.fromarray(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB))

    # 调整 PIL 图片，并转换为 tensor
    img,padding,scale = ImageProcess.normlizeSingleImage(img)
    img = img.to(config.RUN_DEVICE)

    # 模型预测
    with torch.no_grad():
        model.eval()
        predict1,predict2,predict3 = model(img.reshape(1,img.shape[0],img.shape[1],img.shape[2]))

        # 对预测结果解码
        predict1 = BoxProcess.predictionDecode(predict1,config.BOX_ANCHORS[0])
        predict2 = BoxProcess.predictionDecode(predict2,config.BOX_ANCHORS[1])
        predict3 = BoxProcess.predictionDecode(predict3,config.BOX_ANCHORS[2])

        res = PostProcess.detectObjectsFromBatchImages(predict1,predict2,predict3)

        if len(res):
            # 对目标进行监测[[ x1,y1,x2,y2,classScore,classIndex ]] 
            return res[0],padding,scale
        else:
            return None,padding,scale


if __name__ == '__main__':

    model = torch.load('./asset/weightsBackup/yolov3_model_2022_06_03_12_56_50.pth',map_location=torch.device(config.RUN_DEVICE))

    # 均衡化
    clahe = cv2.createCLAHE(2,(8,8))

    # 读取视频
    video = cv2.VideoCapture('./asset/videos/mask.mp4')

    # 打开摄像头
    # video = cv2.VideoCapture(0)

    while video.isOpened():
        # 读取一帧
        flag,frame = video.read()

        if flag != True:
            break
        
        # 缩放图片
        frame = cv2.resize(frame,(0,0),fx=0.5,fy=0.5)

        # 均衡化
        yuv = cv2.cvtColor(frame,cv2.COLOR_BGR2YUV)
        yuv[:,:,0] = clahe.apply(yuv[:,:,0])
        cframe = cv2.cvtColor(yuv,cv2.COLOR_YUV2BGR)

        time_start = time.time()  # 记录开始时间
        # 检测目标
        objs,padding,scale = detectObjsFromFrame(model,cframe)
        if objs is not None:
            frame = ImageProcess.drawBoundingBoxsAndClasses(frame,objs,padding,scale)

        time_end = time.time()  # 记录结束时间
        time_sum = time_end - time_start  # 计算的时间差为程序的执行时间，单位为秒/s
        # 显示帧率
        frame = ImageProcess.drawText(frame,[int(frame.shape[1] * 0.05),int(frame.shape[0] * 0.05)],'FPS: {}'.format( np.round(1/time_sum,2)))

        cv2.imshow('video',frame)
        # cv2.imshow('video1',cframe)

        # 控制播放速度：以 60 帧的速度进行图片显示
        if cv2.waitKey(1) == ord('q'):
            break 
    # 释放
    video.release()
    cv2.destroyAllWindows()

