from turtle import width
import cv2

BC_BLOCK_BOUNDRECT = 0
BC_BLOCK_CONTOUR = 1

def imshow(img,title:str='untitled'):
    """ 显示图片 """
    cv2.imshow(title,img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def sortContoursByRect(contours,isHorizon=True,reverse=False)->list:
    """ 
        按照外接矩形的左上角坐标对轮廓进行排序，并整合外借矩阵
        - isHorizon：true，按照 x 坐标排序；false 安装 y 坐标排序。
        - return：[(boundrect,contour)]
    """
    # 获取外接矩形
    blocks = []
    for i in range(len(contours)):
        rect = cv2.boundingRect(contours[i]) 
        blocks.append((rect,contours[i]))

    # 对轮廓根据外接矩形坐标进行排序：
    if (isHorizon == True):
        blocks = sorted(blocks,key=lambda block:block[BC_BLOCK_BOUNDRECT][0],reverse=reverse)
    else:
        blocks = sorted(blocks,key=lambda block:block[BC_BLOCK_BOUNDRECT][1],reverse=reverse)

    return blocks

def roiByboundRect(src,boundingRect,padding=0):
    """ 根据外接矩形截取图片 """
    xstart,ystart,width, height = boundingRect
    return src[ystart-padding:ystart+padding+height+1,xstart-padding:xstart+width+padding+1]
