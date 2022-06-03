import sys
import os
workPath = os.path.dirname(os.path.dirname(__file__))
if workPath not in sys.path:
    sys.path.append(workPath)

import torch
import Utils.BoxProcess as BoxProcess
from  config import INPUT_SIZE,THRESH_IGNORE,NUM_CLASSFICATIONS,BOX_ANCHORS,LAMBDA_COORD,LAMBDA_CONF,RUN_DEVICE,THRESH_GTBOX_ANCHOR_IOU,LAMBDA_CLASS

class YoloLoss(torch.nn.Module):
    """ yolo v3 损失函数 

        predict: ( bacth, anchor, height,width, [ stx,sty,tw,th ] + stc + s_classfication )
        labels: [[图片编号,分类,x1,y1,x2,y2]]，(x1,y1,x2,y2) 尺寸已经映射到了 INPUT_SIZE
    """
    def __init__(self):
        super(YoloLoss, self).__init__()
        self.mseLoss = torch.nn.MSELoss()
        self.bceLoss = torch.nn.BCELoss()
        self.ceLoss = torch.nn.CrossEntropyLoss()
    
    def forward(self,predict1:torch.Tensor,predict2:torch.Tensor,predict3:torch.Tensor,labels:torch.Tensor):

        loss1,lossvec1 = self.calculateLoss(predict1,labels,BOX_ANCHORS[0])
        loss2,lossvec2 = self.calculateLoss(predict2,labels,BOX_ANCHORS[1])
        loss3,lossvec3 = self.calculateLoss(predict3,labels,BOX_ANCHORS[2])

        return loss1 + loss2 + loss3, lossvec1 + lossvec2 + lossvec3


    def calculateLoss(self,predict:torch.Tensor,labels:torch.Tensor,anchorBoxs:torch.Tensor):
        """计算 predict 预测结果的损失

        Args:
            predict (torch.Tensor): ( bacth, anchor, height,width, [ stx,sty,tw,th ] + stc + s_classfication )
            labels (torch.Tensor): [[图片编号,分类,x1,y1,x2,y2]]，(x1,y1,x2,y2) 尺寸已经映射到了 INPUT_SIZE
            anchorBoxs (torch.Tensor): predict 对应的 anchor box

        Returns:
            loss : 位置损失  + 分类损失 + 样本损失
        """        

        # 感受野
        receptionField = INPUT_SIZE[1] / predict.shape[2] 

        # 定义损失
        lossBox = torch.tensor([0.0],device=RUN_DEVICE)
        lossClass = torch.tensor([0.0],device=RUN_DEVICE)
        lossConf = torch.tensor([0.0],device=RUN_DEVICE)

        # 解析 labels
        target_imgId,target_boxs,obj_indices,obj_class,conf_mask = self.buildTargets(predict,labels,anchorBoxs, receptionField)
    
        target_conf = torch.zeros_like(predict[...,4]).detach()

        if target_imgId is not None:

            mapAnchors,gridx,gridy = obj_indices

            # 与 labels 对应的预测结果
            objPredict = predict[target_imgId, mapAnchors,gridy,gridx]

            # 位置损失
            lossBox = lossBox + self.mseLoss(objPredict[:,0:4], target_boxs )

            # 正样本置信度
            target_conf[target_imgId, mapAnchors,gridy,gridx] = 1

            # 分类
            lossClass = lossClass + self.ceLoss(objPredict[:,5:],obj_class)              

        # Note - 对所有置信度一起计算损失，分开计算效果可能会很差
        lossConf = lossConf + self.bceLoss(predict[conf_mask][:,4],target_conf[conf_mask])

        k = predict.shape[0]
        return k *( lossBox * LAMBDA_COORD  + lossClass * LAMBDA_CLASS + lossConf * LAMBDA_CONF),torch.cat([lossBox,lossClass,lossConf],dim=0).detach()
       

    def buildTargets(self,predict:torch.Tensor,targets:torch.Tensor,anchorBoxs:torch.Tensor,receptionField:float):
        """ 生成期望目标

        Args:
            predict (torch.Tensor): ( bacth, anchor, height,width, [ stx,sty,tw,th ] + stc + s_classfication )
            targets (torch.Tensor): [[图片编号,分类,x1,y1,x2,y2]]，(x1,y1,x2,y2) 尺寸已经映射到了 INPUT_SIZE
            anchorBoxs (torch.Tensor): predict 对应的 anchor box
            receptionField (float): 感受野

        Returns:
            处理后期望目标 
        """        

        # 置信度遮罩，全部的置信度都要参与计算
        conf_mask = torch.ones_like(predict[...,4],dtype=torch.bool,device=RUN_DEVICE).detach()

        # 没有期望目标
        if targets is None:
            return None,None,None,None,conf_mask

        # 形式 [ x1,y1,x2,y2 ] 转 [cx,cy,w,h]
        target_boxs = BoxProcess.box_xy2cwh(targets[:,2:])

        # 查看 ground true box 和 哪个规定的 anchor box 相对应，就是计算 ground true box 与 anchor box 的 iou
        ious = self.anchorBoxAndGTboxsIOU(target_boxs,anchorBoxs)

        # mapAnchors: 根据 ious 重找 ground true box 对应的 anchor box 索引
        max_ious, mapAnchors = torch.max(ious,dim=1)

        # 对 max_ious 进行阈值筛选，防止 ground true box 与 anchor box 的差距过大
        isTargetObj = max_ious > THRESH_GTBOX_ANCHOR_IOU
        mapAnchors = mapAnchors[isTargetObj]

        # ground true box 与当前层的 anchor box 不对应
        if len(mapAnchors) == 0:
            return None,None,None,None,conf_mask
        else:
            target_imgId = targets[isTargetObj,0].long()
            target_class= targets[isTargetObj,1].long()
            target_boxs = target_boxs[isTargetObj]
            # ground true box 相对于 anchor 的比列
            target_boxs[:,2:4] = target_boxs[:,2:4] / anchorBoxs[mapAnchors]
            # 转换为 tw,th
            target_boxs[:,2:4] = torch.log(target_boxs[:,2:4])

            # 根据感受野，将 INPUT_SIZE 尺寸下的 ground true box 缩放到预测特征图的尺寸下 
            target_boxs[:,0:2] = target_boxs[:,0:2] * 1.0 / receptionField
            # [xc,yc] 的缩放结果取整，就是 true box 在预测结果网格中的坐标
            gridxy = target_boxs[:,0:2].long()
            # xc,yc ==> tx,ty
            target_boxs[:,0:2] = target_boxs[:,0:2] - gridxy

            # ============================================ 
            # 找到了  gridH x gridW 的网格中， Anchor box 与 true box 对应位置
            # 计算 正样本 负样本 的标签
            # ============================================ 
            # # 预测网格中的 anchor 有目标
            # mask_obj = torch.zeros((imgN,anchorN,gridH,gridW),dtype=torch.bool)
            # # 正样本
            # mask_obj[target_imgId,label_anchor,trueBoxLocs[:,1],trueBoxLocs[:,0]] = True

            # # 预测网格中的 anchor 没有目标
            # mask_noobj = torch.ones((imgN,anchorN,gridH,gridW),dtype=torch.bool,device=RUN_DEVICE)

            # 去除正样本
            # mask_noobj[target_imgId,mapAnchors,gridxy[:,1],gridxy[:,0]] = False

            # 根据 IOU 阈值，剔除IOU较大的预测值
            if THRESH_IGNORE < 1:
                for i,iou in enumerate(ious[isTargetObj]):
                    # mask_noobj[target_imgId[i],iou > THRESH_IGNORE,gridxy[i,1],gridxy[i,0]] = False
                    conf_mask[target_imgId[i],iou > THRESH_IGNORE,gridxy[i,1],gridxy[i,0]] = False

            # 将 target 对应的标记还原，在上面循环中可能被移除了 
            conf_mask[target_imgId,mapAnchors,gridxy[:,1],gridxy[:,0]] = True

            # 标记出预测结果对应的正确分类，利用 one - hot 进行标记
            obj_class = torch.zeros((len(target_class),NUM_CLASSFICATIONS),device=RUN_DEVICE)
            obj_class[range(len(target_class)),target_class] = 1

            # 预测结果与 ground true box 对应的索引
            obj_indices = (mapAnchors,gridxy[:,0],gridxy[:,1])

        return target_imgId,target_boxs,obj_indices,obj_class,conf_mask


    def anchorBoxAndGTboxsIOU(self,GTboxs:torch.Tensor,anchorBoxs:torch.Tensor):
        """假设 Anchor box 与 ground true box 的中心重合，并计算 Anchor box 与 ground true box 的 IOU

        Args:
            true boxs :  ( N, [cx,cy,w,h] ) ，基于 INPUT_SIZE
            anchor boxs : (NUM_CELL_ANCHOR, [anchor_w , anchor_h]) 基于 INPUT_SIZE

        Return:
            iou : (N,NUM_CELL_ANCHOR)
        """ 
        # (N,1)
        GTboxsArea = (GTboxs[:,2] * GTboxs[:,3]).reshape(-1,1)

        # (1,3)
        anchBoxsArea = (anchorBoxs[:,0] * anchorBoxs[:,1]).reshape(1,-1)

        # (N,3)
        rate = GTboxsArea * 1.0 /  anchBoxsArea 


        # 有可能 GTboxs 比 anchorBox 大，这样比列就大于 1 ，根据 IOU 定义，需要将比值翻转
        ious = torch.where(rate < 1,rate,1/rate)

        return ious



if __name__ == '__main__':

    yoloLoss = YoloLoss()
    # predict1:torch.Tensor,predict2:torch.Tensor,predict3:torch.Tensor,labels

    # predict: ( batch, anchor, height,width, [ bx,by,bw,bh ] + bc + b_classfication ) 其中 [bw,bh] 基于 INPUT_SIZE 尺寸
    # labels: [[图片编号,分类,x1,y1,x2,y2]]，(x1,y1,x2,y2) 尺寸已经映射到了 INPUT_SIZE
    predict1 = torch.rand((1,3,12,12,7))
    predict2 = torch.rand((1,3,6,6,7))
    predict3 = torch.rand((1,3,3,3,7))

    labels = torch.tensor([[0,1,12,19,20,20],[0,0,1,1,40,40]])

    lo =  yoloLoss(predict1,predict2,predict3,labels)    
