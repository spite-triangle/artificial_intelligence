import dlib
import numpy as np
import  cv2

def preprocess(path,fx=0.5,fy=0.5):
    img = cv2.imread(path)
    img = cv2.resize(img, (0,0),fx=fx,fy=fy)
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return (img,imgGray)

def imshow(img,title='untitled'):
    cv2.imshow(title, img)
    cv2.waitKey(0)

def lableFaces(canvas,facesLocs):
    for face in facesLocs:
        y1 = face.bottom()  # detect box bottom y value
        y2 = face.top()  # top y value
        x1 = face.left()  # left x value
        x2 = face.right()  # right x value
        cv2.rectangle(canvas,(x1,y1),(x2,y2),(0,0,255),2)

def facesKeypointDescritptions(img,imgGray,facesLocs,predictor,encoder,jet=1):
    # 特征点位置
    keypointsLocs = [predictor(img,faceLoc) for faceLoc in facesLocs]

    # 获取描述符
    return np.array([encoder.compute_face_descriptor(img,keypointsLoc,jet) for keypointsLoc in keypointsLocs])

if __name__ == '__main__':
    # 载入图片
    facesImg,facesImgGray = preprocess('./asset/faces.jpg')
    targetImg,targetImgGray = preprocess('./asset/mads1.png')

    # 人脸检测器
    detector = dlib.get_frontal_face_detector()

    # 特征点预测器
    predictor = dlib.shape_predictor('./asset/shape_predictor_68_face_landmarks.dat')

    # 特征描述生成模型
    encoder = dlib.face_recognition_model_v1('./asset/dlib_face_recognition_resnet_model_v1.dat')
    
    #  标定人脸位置
    facesLocs = detector(facesImgGray,0)
    targetLocs = detector(targetImgGray,0)

    # 获取人脸特征描述
    facesDescriptions = facesKeypointDescritptions(facesImg,facesImgGray, facesLocs, predictor, encoder)
    targetDescription = facesKeypointDescritptions(targetImg,targetImgGray, targetLocs, predictor, encoder)

    print(facesDescriptions.shape)
    print(targetDescription.shape)

    # 描述符对比，计算欧氏距离
    distances = np.linalg.norm(facesDescriptions - targetDescription,axis=1)

    print(np.argmin(distances))

    # 将结果标记出来
    lableFaces(facesImg, [facesLocs[np.argmin(distances)]])

    lableFaces(targetImg, targetLocs)

    imshow(facesImg)
    imshow(targetImg)
