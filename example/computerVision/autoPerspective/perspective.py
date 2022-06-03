import cv2
import utils
import numpy as np

def getOutline(img):
    """ 得到纸张的轮廓 """
    # 预处理，找图像边缘
    imgGray = cv2.cvtColor(imgSrc, cv2.COLOR_BGR2GRAY)
    ratio = 500 / imgSrc.shape[0] 
    imgGray = cv2.resize(imgGray,(0,0),fx=ratio,fy=ratio)  
    imgGray = cv2.GaussianBlur(imgGray,(3,3),sigmaX=4)
    imgEdge = cv2.Canny(imgGray,190,220)
    # utils.imshow(imgEdge)

    # 提取轮廓
    contours = cv2.findContours(imgEdge,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[1]
    # 对轮廓面积进行排序，面积最大的一定是纸张的轮廓
    contours = sorted(contours,key=cv2.contourArea,reverse=True)

    # cv2.drawContours(imgGray,contours,0,(0,0,255),2) 
    # utils.imshow(imgGray)

    return np.array(contours[0]/ratio,dtype=np.int)

def getQuadPointByCnt(cnt):
    """ 获取四边形外轮廓的四个点 """
    # 获取顶点
    epsilon = 0.01 * cv2.arcLength(cnt,True)
    cntApprox = cv2.approxPolyDP(cnt,epsilon,closed=True)

    # 区分四个顶点位置，存放顺序从左上角开始，逆时针
    # 按照 x 方向的大小对四个点排序
    cntApprrox = sorted(cntApprox,key=lambda point:point[0][0])

    # 左边两点的高低
    if cntApprox[0][0][1] > cntApprox[1][0][1]:
        temp = list(cntApprox[0][0])
        cntApprox[0][0] = list(cntApprox[1][0])
        cntApprox[1][0] = temp

    # 右边两点的高低
    if cntApprox[2][0][1] < cntApprox[3][0][1]:
        temp = list(cntApprox[2][0])
        cntApprox[2][0] = list(cntApprox[3][0])
        cntApprox[3][0] = temp
    
    return cntApprox


if __name__ == '__main__':
    imgSrc = cv2.imread('./perspectivePaper.jpg')

    # 获取paper的外轮廓
    paperOutline = getOutline(imgSrc)

    # 提取四个顶点
    quadCnt = getQuadPointByCnt(paperOutline)


    # 透视变换
    points = np.array(quadCnt,dtype=np.float32).reshape((4,-1))
    width =  np.sqrt((points[0,0] - points[3,0])**2 + (points[0,1] - points[3,1])**2) 
    height =  np.sqrt((points[0,0] - points[2,0])**2 + (points[0,1] - points[2,1])**2) 
    width = int(width)
    height = int(height)
    dstPoints = np.array([[0,0],[0,height],[width,height],[width,0]],dtype=np.float32)
    m = cv2.getPerspectiveTransform(points,dstPoints)
    imgr = cv2.warpPerspective(imgSrc,m,(width,height))

    utils.imshow(imgr)
    # cv2.drawContours(imgSrc,[quadCnt],0,(0,0,255),2) 
    # utils.imshow(imgSrc)

