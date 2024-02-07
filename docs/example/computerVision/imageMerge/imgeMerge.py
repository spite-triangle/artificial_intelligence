import cv2
import utils
import numpy as np

def siftDetectAndCompute(img):
    """ 关键点、描述符号提取 """
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    sift = cv2.SIFT_create()
    kps,des = sift.detectAndCompute(imgGray,None)

    return kps,des

def knnMatch(des1,des2):
    # 建立匹配算法
    bf = cv2.BFMatcher()

    # 匹配
    matchRes = bf.knnMatch(des1,des2,k=2)

    goodMatchs=[]
    for m, n in matchRes:
        if m.distance < 0.75*n.distance:
            goodMatchs.append(m) 

    return goodMatchs 

if __name__ == '__main__':
    img1 = cv2.imread('./asset/part1.jpg')
    img2 = cv2.imread('./asset/part2.jpg')

    # 计算特征
    kps1,des1 = siftDetectAndCompute(img1)
    kps2,des2 = siftDetectAndCompute(img2)

    # 匹配
    matchs = knnMatch(des1,des2)

    # 至少 4 对才能计算出透视变换矩阵；RANSAC算法采样还要多要一点
    if (len(matchs) > 4):

        srcPoints = []
        dstPoints = []
        # 配对点的位置
        for match in matchs:
            srcPoints.append(kps1[match.queryIdx].pt)
            dstPoints.append(kps2[match.trainIdx].pt)

        # 类型转换
        srcPoints = np.float32(srcPoints)
        dstPoints = np.float32(dstPoints)

        # ransacReprojThreshold：区分内点与外点的阈值
        # findHomography(srcPoints, dstPoints[, method[, ransacReprojThreshold[, 
        # mask[, maxIters[, confidence]]]]]) -> retval, mask
        h,mask =  cv2.findHomography(srcPoints, dstPoints,cv2.RANSAC,4)

        # 合并图片
        imgres = cv2.warpPerspective(img1,h,(img1.shape[1] + img2.shape[1],max(img1.shape[0],img2.shape[0])))
        imgres[0:img2.shape[0],0:img2.shape[1]] = img2

        utils.imshow(imgres)
