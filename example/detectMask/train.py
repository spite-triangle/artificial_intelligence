import os
import torch
import ManageData.Datasets as Datasets
import Model.Network as Network
import Model.Loss as Loss
import Utils.BoxProcess as BoxProcess
import Utils.PostProcess as PostProcess
import config
import time

def removeRedundantPth(pthDir:str,retain:int=3):
    """按照创建时间，清理多余的 pth 模型文件。模型太多占用空间，最终导致程序崩溃。

    Args:
        pthDir (str): 模型存放的路径
        retain (int, optional): 清理时，保留多少个模型文件. Defaults to 3.
    """    
    if  not os.path.isdir(pthDir):
        print('\033[33m [warnning directory] \033[0m', 'It\'s not a right directory about \'{}\''.format(pthDir))
        return

    files = os.listdir(pthDir)
    if not files or len(files) <= retain :
        return
    else:
        # os.path.getmtime() 函数是获取文件最后修改时间
        # os.path.getctime() 函数是获取文件最后创建时间
        files = sorted(files,key=lambda x: os.path.getmtime(os.path.join(pthDir, x)))

        # 移除多余文件
        for item in range(len(files) - retain):
            os.remove(os.path.join(pthDir, files[item]))


def learnRateAttenuation(epoch:int):
    """learning rate 的衰减倍数

    Args:
        epoch (int): 迭代次数

    """    
    w = 1.0 / (1.0 + config.DELAY_RATE * epoch)
    w = max(w,10e-4)
    return w


if __name__ == '__main__':
    # 导入数据
    trainDataset = Datasets.ImageLabelDataset('./asset/trainSet',imageFolder='train_images',labelFolder='train_labels')
    testDataset = Datasets.ImageLabelDataset('./asset/testSet',imageFolder='test_images',labelFolder='test_labels')

    # 数据加载器
    trainDataLoader = torch.utils.data.DataLoader(trainDataset,config.BATCH_SIEZE,shuffle=True,collate_fn=trainDataset.collect_fn,drop_last=True,num_workers=3)
    testDataLoader = torch.utils.data.DataLoader(testDataset,config.BATCH_SIEZE,shuffle=True,collate_fn=trainDataset.collect_fn,drop_last=True,num_workers=3)

    # 模型
    # model = Network.Darknet53()
    # model.to(config.RUN_DEVICE)
    # model.load_state_dict(torch.load('./asset/weights/yolov3_model_59_100.pth'))
    model = torch.load('./asset/weightsBackup/yolov3_model_2022_06_03_12_56_50.pth',map_location=torch.device(config.RUN_DEVICE))

    # 损失函数
    lossFcn = Loss.YoloLoss()
    lossFcn.to(config.RUN_DEVICE)

    # 优化器
    optimizer = torch.optim.Adam(model.parameters(),lr=config.LEARN_RATE)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,lr_lambda=learnRateAttenuation)


    for epoch in range(config.EPOCHS):
        print("===================== epoch: {} ===========================".format(epoch))
        
        #+++++++++++++++++
        # 训练部分
        #+++++++++++++++++
        batchCount = 0
        lossSum = 0.0
        lossvecSum = torch.tensor([0.,0.,0.],device=config.RUN_DEVICE)

        for images,lables in trainDataLoader:
            images = images.to(config.RUN_DEVICE)
            lables = lables.to(config.RUN_DEVICE)

            # 启动训练模型
            model.train()
            predict1,predict2,predict3 = model(images)

            # 计算损失
            loss,lossvec =lossFcn(predict1,predict2,predict3,lables)

            # 清除梯度
            optimizer.zero_grad()

            # 反向传播
            loss.backward()

            # 更新梯度与学习率
            optimizer.step()
            scheduler.step() 

            # 打印损失
            batchCount = batchCount + 1
            if batchCount % 50 == 0:
                print("[{}] batch, loss: {} , position loss : {}, class loss : {} , conf losss: {}".format(batchCount,loss.item(),lossvec[0],lossvec[1],lossvec[2]))

                # torch.save(model.state_dict(),'./asset/weights/yolov3_model_{}_{}.pth'.format(epoch,batchCount))
                torch.save(model,'./asset/weights/yolov3_model_epoch{}_{}_{}.pth'.format(epoch,batchCount,time.strftime('%Y_%m_%d_%H_%M_%S', time.localtime())))

            lossSum = lossSum + loss.item()
            lossvecSum = lossvecSum + lossvec

        lossvecSum = lossvecSum / len(trainDataLoader)
        print("summary loss: {},  position loss : {}, class loss : {} ,conf loss : {}".format(lossSum / len(trainDataLoader),lossvecSum[0],lossvecSum[1],lossvecSum[2]))


        #+++++++++++++++++
        # 测试部分
        #+++++++++++++++++
        metrics = []
        for images,lables in testDataLoader:
            images = images.to(config.RUN_DEVICE)
            lables = lables.to(config.RUN_DEVICE)

            with torch.no_grad():
                # 启动测试模型
                model.eval()

                predict1,predict2,predict3 = model(images)

                # 对预测结果解码
                predict1 = BoxProcess.predictionDecode(predict1,config.BOX_ANCHORS[0])
                predict2 = BoxProcess.predictionDecode(predict2,config.BOX_ANCHORS[1])
                predict3 = BoxProcess.predictionDecode(predict3,config.BOX_ANCHORS[2])

                # 对目标进行监测
                objsPerImages = PostProcess.detectObjectsFromBatchImages(predict1,predict2,predict3)

                res = PostProcess.evaluate(objsPerImages,lables,config.THRESH_AP_IOU)

                metrics.append(res)

        
        # 打印测试指标
        metrics = torch.cat(metrics,dim=0)
        #  [[classIndex,precision,recall,ap]]
        res = metrics[:,1:].sum(dim=0) / len(metrics)
        print("precision: {}\nrecall: {}\nap{}: {}".format(res[0],res[1],config.THRESH_AP_IOU*100,res[2]))

        # 清理多余的 pth 文件
        if epoch % 2 == 0:
            removeRedundantPth('./asset/weights')

