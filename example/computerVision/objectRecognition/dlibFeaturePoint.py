import cv2
import dlib
import  numpy as np



img = cv2.imread('./asset/face.png')
img = cv2.resize(img,(0,0),fx=0.5,fy=0.5)
imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 人脸检测器
detector = dlib.get_frontal_face_detector()

# 找到人脸
face:dlib.rectangle = detector(img,0)[0]

# 展示人脸
faceImg = img[face.top() :face.bottom(),face.left() :face.right()]
cv2.imshow('face', faceImg)

# 加载关键点预测器
predictor:dlib.shape_predictor = dlib.shape_predictor('./asset/shape_predictor_68_face_landmarks.dat')

# 预测关键点
points : dlib.full_object_detection = predictor(img,face)

for i in range(len(points.parts())):
    # 点
    point:dlib.point = points.part(i)

    # 绘制点
    cv2.circle(img, (point.x,point.y), 2, (0,255,0),1)


# 将点全部提取出来
pts = np.array([( point.x,point.y )for point in points.parts()],dtype=np.int32)

cv2.polylines(img, [pts[36:42]], True,(255,0,0),3)

cv2.imshow('key points', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
