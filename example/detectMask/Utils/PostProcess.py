import sys
import os

workPath = os.path.dirname(os.path.dirname(__file__))
if workPath not in sys.path:
    sys.path.append(workPath)

import torch
from config import INPUT_SIZE, THRESH_OBJ,THRESH_NMS,THRESH_SCORE,THRESH_BOX_MIN_SIZE,THRESH_BOX_MAX_NUM,RUN_DEVICE,NUM_CLASSFICATIONS
import Utils.BoxProcess as BoxProcess

import  matplotlib.pyplot as plt

def evaluate(objsPerImgs:list,labels:torch.Tensor,threshIOU:float):
    """对预测结果进行评估

    Args:
        objsPerImgs (list): 每张图片中检测出来的目标
        labels (torch.Tensor): 标记的真实结果
        threshIOU (float): 用于划分正确判断的阈值

    Returns:
        metrics : 评价指标 [[classIndex,precision,recall,ap]]
    """    

    if len(objsPerImgs) == 0:
        return torch.zeros((1,4),device=RUN_DEVICE)

    # 统计出预测结果中，哪些预测正确了
    correntMark = predictionStatistics(objsPerImgs,labels,threshIOU)

    # 根据预测的统计结果，计算各个指标
    metrics = calculateMetrics(correntMark,labels)

    return metrics

def predictionStatistics(objsPerImgs:list,labels:torch.Tensor,threshIOU:float) -> torch.Tensor:
    """预测结果与 labels 中的真实结果进行对比，标记出预测正确的结果

    Args:
        objsPerImgs (list):  每张图片中检测出来的目标，[ torch.Tensor ]，其中的 tensor 为 [[ x1,y1,x2,y2,classScore,classIndex ]]
        labels (torch.Tensor): [[图片编号,分类,x1,y1,x2,y2]]
    Return:
        correntMark:  [[classScore,classIndex, predictTrue]]
    """    
    correntMark= []
    # 对每张图片的预测结果进行遍历
    for imgID,objBoxs in enumerate(objsPerImgs):

        # 当前图片每一预测结果
        if(objBoxs is None):
            continue

        # 获取图片对应的 label
        #  [分类,x1,y1,x2,y2]
        targetBoxs = labels[labels[:,0] == imgID][:,1:]

        # 目标正确检测的标记
        predictTrue = torch.zeros(len(objBoxs),dtype=torch.int,device=RUN_DEVICE) 

        # 检测图片中存在目标
        if len(targetBoxs) != 0:
            # 记录已经被检测过的 target
            hasDetected = []

            # 将预测的 box 与 targetboxs 一一对比
            for index, objBox in enumerate(objBoxs):

                # target 对应的预测框全部找到了 
                if len(hasDetected) == len(targetBoxs):
                    break

                # 当前 box 的分类
                objClass = objBox[-1]

                # 当前 box 分类在 targetBoxs 里面不存在
                if objClass not in targetBoxs[:,0]:
                    continue

                # 获取 targetBox 中对应的分类
                # filteredTargets：从 targetBoxs 中获取所有 objClass 类型
                # targetIndices：filteredTargets 对应 targetBoxs 的索引
                targetIndices,filteredTargets = zip(*filter(lambda x: targetBoxs[x[0],0] == objClass ,enumerate(targetBoxs)))

                # 上面过滤后，返回的是 [torch.Tensor] 类型，需要转换为 Tensor
                filteredTargets = torch.cat(filteredTargets,dim=0).reshape(-1,5)

                # 计算当前 objBox 与 classTargetes 对应的 IOU
                # 找到最大 iou
                ious = BoxProcess.bbox_iou(objBox[0:4],filteredTargets[:,1:]).reshape(-1,1)
                iou_max, iou_index =  ious.max(0)
                targetIndex = targetIndices[iou_index]
                
                # 当前 box 与 target 对应的最大 iou 是否达到阈值
                # 当前的 iou_max 对应的 target 是否已经找到
                if iou_max > threshIOU and targetIndex not in hasDetected:
                    hasDetected.append(targetIndex)
                    predictTrue[index] = 1

        # 组织输出结果 [[ classScore,classIndex, predictTrue]]
        correntMark.append(torch.cat([ objBoxs[:,4:],predictTrue.reshape(-1,1) ],dim=1))

    return torch.cat(correntMark,dim=0)

def calculateMetrics(correntMark:torch.Tensor,labels:torch.Tensor):
    """对多个类别的目标，分别计算 AP

    Args:
        correntMark(torch.Tensor): [[ classScore,classIndex, predictTrue]]
        labels (torch.Tensor): [[图片编号,分类,x1,y1,x2,y2]]
    Return:
        metrics (torch.Tensor) : [[classIndex,precision,recall,ap]]
    """ 

    # 所有 target 目标的分类
    targetClass = labels[:,1]
    
    # 对 correntMark 按照 classScore 进行排序
    correntMark= correntMark[ correntMark[:,0].argsort(descending=True) ]

    # class,pricision,recall,ap
    metrics = []
    # 每个类别分别计算 AP
    for c in targetClass.unique():
        
        # 统计 c 类别在预测结果中的总数
        nPredict = (correntMark[:,1] == c).sum()

        # 统计 c 类别在 targetClass 中的总数
        nTarget = (targetClass == c).sum()

        # c 在预测结果 或者 target 中不存在
        if nPredict == 0 or nTarget == 0:
            metrics.append([c,0,0,0])
            continue

        # 当前类别的 correntMark
        ccorrentMark= correntMark[correntMark[:,1]==c]
        
        # true positive cumulative sum
        tp = ccorrentMark[:,2].cumsum(0)

        # false positve cumulative sum
        fp = (1 - ccorrentMark[:,2]).cumsum(0)

        # precision 序列
        precisionCurve = tp / ( tp + fp)

        # recall 序列
        recallCurve = tp / ( nTarget + + 1e-16)

        ap = calculateAP(precisionCurve,recallCurve) 
       
        # c 分类的 precision
        metrics.append([c,precisionCurve[-1],recallCurve[-1],ap])

    return torch.tensor(metrics,device=RUN_DEVICE)

def calculateAP(precisionCurve:torch.Tensor,recallCurve:torch.Tensor):
    """根据 precisionCurve 与 recallCurve 计算 ap 值

    Args:
        precisionCurve (torch.Tensor): 准确度曲线
        recallCurve (torch.Tensor): 召回率曲线
    Return:
        ap ：ap 值
    """    

    precisionCurve = torch.cat([torch.tensor([0.0],device=RUN_DEVICE),precisionCurve,torch.tensor([0.0],device=RUN_DEVICE)])
    recallCurve = torch.cat([torch.tensor([0.0],device=RUN_DEVICE),recallCurve,torch.tensor([1.0],device=RUN_DEVICE)])

    # 用于之后图像绘制
    # precisionCurve1 = precisionCurve.clone()
    # recallCurve1 = recallCurve.clone()

    # 对 precision 进行近似
    for i in range(len(precisionCurve)-1,0,-1):
        precisionCurve[i - 1] = torch.maximum(precisionCurve[i],precisionCurve[i-1])

    # 查找 recall 值改变的位置
    index = torch.where(recallCurve[1:] != recallCurve[:-1])[0]

    # recallCurve[index]    recall 突变前的值
    # recallCurve[index+1]  recall 突变后的值
    ap = torch.sum((recallCurve[index+1] - recallCurve[index]) * precisionCurve[index + 1]) 

    #++++++++++++++++++++++++++++++++++++++++++
    # 绘图
    #++++++++++++++++++++++++++++++++++++++++++
    # print(index)
    # plt.subplot(211)
    # plt.plot(recallCurve1.tolist(),precisionCurve1.tolist(),'-o')
    # plt.subplot(212)
    # plt.plot(recallCurve.tolist(),precisionCurve.tolist(),'-o')
    # plt.show()

    return ap


def detectObjectsFromBatchImages(predict1,predict2,predict3):
    """ 对多张图片中的目标进行检测

    Args:
        predict1 (torch.Tensor): 模型预测结果 ( anchor, height,width, [ bx,by,bw,bh ] + bc + b_classfication )
        predict2 (torch.Tensor): 模型预测结果
        predict3 (torch.Tensor): 模型预测结果 

    Returns:
        list : 每张图片中检测出来的目标，[ torch.Tensor ]，其中的 tensor 为 [[ x1,y1,x2,y2,classScore,classIndex ]]
    """    
    # 每张图片中检测出的目标
    objsPerImgs = []
    # 遍历每张图片的预测结果
    for i in range(len(predict1)):
        # 对一张图片进行目标检测，并将检测结果存放到 list 中 
        objs = detectObjectsFromSingleImage(predict1[i],predict2[i],predict3[i])
        if objs is not None:
            objsPerImgs.append(objs) 

    return objsPerImgs

def detectObjectsFromSingleImage(predict1,predict2,predict3):
    """ 单张图片的预测结果进行目标检测

    Args:
        predict : ( anchor, height,width, [ bx,by,bw,bh ] + bc + b_classfication )
    其中 [ bx,by,bw,bh] 基于 INPUT_SIZE 尺寸

    Returns:
        检测到的目标 : [[ x1,y1,x2,y2,classScore,classIndex ]]
    """ 

    # Note - 将三层的预测结果一起进行 「极大值抑制」
    predictAll = torch.cat([predict1.reshape(-1,5 + NUM_CLASSFICATIONS ),
                        predict2.reshape(-1,5 + NUM_CLASSFICATIONS),
                        predict3.reshape(-1,5 + NUM_CLASSFICATIONS)],dim=0)

    # 进行极大值抑制
    out = non_max_suppression(predictAll)
    
    # 判断是否非空
    if out is not None:
        return out
    return None

def non_max_suppression(prediction):
    """ 对预测的框进行

    Args:
        prediction :  ( anchor, height,width, [ bx,by,bw,bh ] + bc + b_classfication ) 
                        其中 [bw,bh] 基于 INPUT_SIZE 尺寸
    """    
    # 取出存在目标的框。 ( n_box ,[ bx,by,bw,bh ] + bc + b_classfication )
    boxs = prediction[ prediction[...,4] > THRESH_OBJ] 

    # 计算 score = max(b_classfication) * bc  之后还要根据 b_classfication 中最大的进行分类，所以只算最大的
    boxs[:,4] = boxs[:,5:].max(dim=1)[0]  * boxs[:,4]

    # 对 score 进行过滤；对预测框的最小尺寸进行限制
    boxs = boxs[ (boxs[:,4] > THRESH_SCORE) 
                & (boxs[:,2:4] > THRESH_BOX_MIN_SIZE ).all(dim=1) ]

    # 预测结果中没有目标
    if len(boxs) == 0:
        return None

    # 分类置信度，分类的索引
    classIndex = boxs[:,5:].argmax(dim=1)
    classScore = boxs[:,4]

    # 更改 box 形状由 [ bx,by,bw,bh ] 换算为 [x1,y1,x2,y2]
    boxs[:,:4] = BoxProcess.box_cwh2xy(boxs[:,:4]) 

    # 对出界的矩形框进行限制
    # boxs[:,[0,2]] = boxs[:,[0,2]].clamp_(0,INPUT_SIZE[0] - 1)
    # boxs[:,[1,3]] = boxs[:,[1,3]].clamp_(0,INPUT_SIZE[1] - 1)

    # 重组结果，得到 [x1,y1,x2,y2,classScore,classIndex]
    boxs = torch.cat((boxs[:,:4],classScore.reshape(-1,1),classIndex.reshape(-1,1)),dim=1)

    # 根据 score 对预测框进行降序排序
    boxs = boxs[ boxs[:,4].argsort(descending=True) ]

    # 目标框
    objBoxs = []
    # 对不同的分类分别进行极大值抑制
    for c in boxs[:,-1].unique():
        # c 分类对应的 box
        cboxs = boxs[ boxs[:,-1] == c ]

        # 预测框限制
        if len(cboxs) == 1:
            objBoxs.append(cboxs[0])
            break
        elif len(cboxs) > THRESH_BOX_MAX_NUM:
            cboxs = cboxs[:THRESH_BOX_MAX_NUM] 

        # 极大抑制
        while len(cboxs) :
            # 第一个 box 的 score 最大，作为基准框
            objBoxs.append(cboxs[0])

            # 只剩最后一个了，不再计算
            if len(cboxs) == 1:
                break

            # 第一个基准框与其他框计算 IOU
            iou = BoxProcess.bbox_iou(cboxs[0],cboxs[1:])

            # iou 大于抑制阈值的框，全部去除
            cboxs = cboxs[1:][iou < THRESH_NMS]

    if len(objBoxs):
        return torch.vstack(objBoxs)
    else:
        return None

if __name__ == "__main__":
    predict = torch.zeros((3,2,2,7))

    # 目标
    predict[0,0,0] = torch.tensor([5,5,11,11,0.9,0.9,0.2])
    predict[1,1,0] = torch.tensor([5,4,11,12,0.8,0.8,0.2])
    predict[0,1,0] = torch.tensor([8,7,11,13,0.9,0.7,0.6])
    predict[2,1,0] = torch.tensor([3,5,11,14,0.9,0.2,0.6])
    predict[1,1,1] = torch.tensor([15,5,11,15,0.1,0.7,0.5])
    box =  detectObjectsFromSingleImage([predict])


    # 测试指标计算
    correntMark = torch.tensor([[0.23,1,0],
                                [0.45,1,1],
                                [0.66,3,1],
                                [0.34,2,1],
                                [0.55,1,0],
                                [0.32,2,0],
                                [0.71,2,1],
                                [0.89,1,1],
                                [0.76,3,1],
                                [0.43,1,0]])
    labels = torch.tensor([[1,1],
                            [2,1],
                            [3,1],
                            [4,2],
                            [5,2],
                            [6,2],
                            [7,3],
                            [8,3],
                            [9,4]],dtype=torch.float)
    metrics = calculateMetrics(correntMark,labels)

    # 测试正确结果统计
    objsPerImgs = [
        torch.tensor([[2,2,11,11,0.6,1],
                    [1,1,16,16,0.4,2],
                    [5,5,20,20,0.7,1],
                    [8,8,15,15,0.5,2],
                    [8,8,15,15,0.5,1],
                    [0,0,15,15,0.8,1],
                    [0,0,15,15,0.8,3]]),
        torch.tensor([[2,3,15,15,0.8,1]])
    ]

    labels = torch.tensor([[0,2,0,0,14,14],
                            [0,1,7,7,14,14],
                            [0,1,5,5,15,15],
                            [1,1,5,5,12,12]])

    correntMark = predictionStatistics(objsPerImgs,labels,0.5)

    # 评价
    metrics = evaluate(objsPerImgs,labels,0.5) 