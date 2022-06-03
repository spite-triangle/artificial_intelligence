import torch
from PIL import Image
import Utils.BoxProcess as BoxProcess
import Utils.ImageProcess as ImageProcess
import Utils.PostProcess as PostProcess
import Model.Network as Network
import config



def detectObjsFromImages(model:Network.Darknet53,images:torch.Tensor):
    """从一张图片中检测出目标物体

    Args:
        model (Network.Darknet53): yolo v3 模型
        images (torch.Tensor): 图片 tensor  ,(batch,channel,height,width)

    Returns:
        objsPerImages : 目标检测结果 [[ x1,y1,x2,y2,classScore,classIndex ]] 
    """    

    # 模型预测
    with torch.no_grad():
        model.eval()
        predict1,predict2,predict3 = model(images)

        # 对预测结果解码
        predict1 = BoxProcess.predictionDecode(predict1,config.BOX_ANCHORS[0])
        predict2 = BoxProcess.predictionDecode(predict2,config.BOX_ANCHORS[1])
        predict3 = BoxProcess.predictionDecode(predict3,config.BOX_ANCHORS[2])

        # 对目标进行监测[[ x1,y1,x2,y2,classScore,classIndex ]] 
        objsPerImages = PostProcess.detectObjectsFromBatchImages(predict1,predict2,predict3)
        return objsPerImages


if __name__ == '__main__':

    # 加载图片
    image = Image.open('./asset/images/mads.jpg').convert('RGB')

    img,padding,scale = ImageProcess.normlizeSingleImage(image)
    img = img.to(config.RUN_DEVICE)

    # 加载模型
    # model = Network.Darknet53()
    # model.to(config.RUN_DEVICE)
    # model.load_state_dict(torch.load('./asset/yolov3_model_52_100.pth',map_location=torch.device(config.RUN_DEVICE)))
    model = torch.load('./asset/weightsBackup/yolov3_model_2022_06_03_12_56_50.pth',map_location=torch.device(config.RUN_DEVICE))
    
    # 目标检测
    objsPerImages = detectObjsFromImages(model,img.reshape(1,img.shape[0],img.shape[1],img.shape[2]))

    if len(objsPerImages):
        # 绘制矩形框
        img = ImageProcess.drawBoundingBoxsAndClasses(image,objsPerImages[0],padding,scale)
        ImageProcess.showCV(img)
