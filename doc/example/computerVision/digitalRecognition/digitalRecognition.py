import cv2
import utils 
import numpy as np

def getDigtalsRoi(img,reverse=False):
    """ 获取数字部分图像 """
    # 转灰度图
    imggray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    # 二值化
    if reverse == True:
        retval,imgBinary = cv2.threshold(imggray,0,255,cv2.THRESH_BINARY_INV|cv2.THRESH_OTSU)
    else:
        retval,imgBinary = cv2.threshold(imggray,0,255,cv2.THRESH_BINARY|cv2.THRESH_OTSU)

    # 提取最外侧轮廓
    imgsrc, contours, hierarchy = cv2.findContours(imgBinary,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

    # 轮廓排序
    blocks = utils.sortContoursByRect(contours)

    # 根据外接矩形截取图片
    digitals = []
    for i in range(len(blocks)):
        roi = utils.roiByboundRect(imgBinary,blocks[i][utils.BC_BLOCK_BOUNDRECT])
        roi = cv2.resize(roi,(55,80))
        digitals.append(roi)

    return digitals


def getSrcDigitalBlocks(img):
    """ 获取输入图片的数字块轮廓 """
    # 转灰度图
    widthSrc = img.shape[1]
    heightSrc = img.shape[0]
    img = cv2.resize(img,(293,185))
    imggray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    # 二值化
    retval,imgBinary = cv2.threshold(imggray,0,255,cv2.THRESH_BINARY|cv2.THRESH_OTSU)
    # utils.imshow(imgBinary)

    # 去掉一些大的部分
    kernelMorp = cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))
    imgTop = cv2.morphologyEx(imgBinary,cv2.MORPH_TOPHAT,kernelMorp,iterations=1)
    # utils.imshow(imgTop)

    # imgSobel = cv2.Sobel(imgTop,cv2.CV_16S,dx=1,dy=0,ksize=5)
    # imgSobel = (np.abs(imgSobel) / imgSobel.max() ) * 255
    # utils.imshow(imgSobel)

    # 将所有形状连成块
    kernelMorp = cv2.getStructuringElement(cv2.MORPH_RECT,(9,3))
    imgClose = cv2.morphologyEx(imgTop,cv2.MORPH_CLOSE,kernelMorp,iterations=2)
    # utils.imshow(imgClose)

    imgClose = cv2.resize(imgClose,(widthSrc,heightSrc))

    # 提取最外侧轮廓
    imgsrc, contours, hierarchy = cv2.findContours(imgClose,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    # 根据外接矩形比列筛选轮廓
    digitalsBlockCnts = []
    for contour in contours:
        startx,starty,width,height = cv2.boundingRect(contour)
        rate = width * 1.0  / height
        area = cv2.contourArea(contour)
        if (area > 1000 and rate > 3.1 and rate < 4):            
            digitalsBlockCnts.append(contour)

    # imgSrc = cv2.imread('./asset/card1.jpg')
    canva = cv2.merge([imgClose,imgClose,imgClose])
    imgContour = cv2.drawContours(canva,digitalsBlockCnts,-1,(0,0,200),2)
    utils.imshow(canva)

    # 对轮廓排序
    return utils.sortContoursByRect(digitalsBlockCnts)



if __name__ == '__main__':
    # 读取图片
    imgSrc = cv2.imread('./asset/card.jpg')
    imgTemp = cv2.imread('./asset/template.jpg')

    # 模板预处理
    digitalsTemp = getDigtalsRoi(imgTemp,reverse=True)

    # 原图预处理
    digitalBlocks = getSrcDigitalBlocks(imgSrc)

    # 解析数字
    digitals = ''
    for boundRect,contour in digitalBlocks:
        imgBlock = utils.roiByboundRect(imgSrc,boundRect)
        imgDigitals = getDigtalsRoi(imgBlock)

        # 遍历每一个输入图片的数字
        for digital in imgDigitals:
            num = -1
            score = 10
            # 与模板进行对比
            for i in range(len(digitalsTemp)):
                res = cv2.matchTemplate(digital,digitalsTemp[i],cv2.TM_SQDIFF_NORMED)
                res = res.ravel()[0]
                if res < score :
                    num = i
                    score = res
            digitals = digitals + str(num)

    print(digitals)