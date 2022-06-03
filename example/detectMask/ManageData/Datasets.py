import os
import sys
import torch
from PIL import  Image
import Utils.BoxProcess as BoxProcess
import Utils.ImageProcess as ImageProcess
from config import BATCH_SIEZE

class ImageLabelDataset(torch.utils.data.Dataset):
    """数据集的目录树：
        - root
        -- imageFolder
        -- labelFolder
    labels: [[图片编号,分类,x1,y1,x2,y2]]，(x1,y1,x2,y2) 大小映射到了 INPUT_SIZE
    """ 

    def __init__(self,root:str,imageFolder:str='train_images',labelFolder:str='train_labels',labelSuffix:str='.xml'):
        """
        Args:
            root (str):INPUT_SIZEINPUT_SIZE
            imageFolder (str, optional): 图片文件夹名. Defaults to 'train_images'.
            labelFolder (str, optional): 标签文件夹名. Defaults to 'test_labels'.
            labelSuffix (str, optional): 标签文件后缀. Defaults to '.xml'.
        """

        self.imageFolderPath = os.path.join(root,imageFolder).replace('\\','/')
        self.labelFolderPath = os.path.join(root,labelFolder).replace('\\','/')
        self.imageFileNames = os.listdir(self.imageFolderPath) 
        self.labelSuffix = labelSuffix

    def __getitem__(self,index):
        """ 获取一项数据 """
        try:
            # 读取图片
            imageName = self.imageFileNames[index]
            imagePath = self.imageFolderPath + '/' + imageName
            image = Image.open(imagePath)
            image,padding,scaleRate = ImageProcess.normlizeSingleImage(image)
        except Exception as e:
            print('\033[31m [error image] \033[0m','path: '+ imagePath)
            print(e)
            sys.exit()


        try:
            # 读取label
            labelName = imageName.split('.')[0] + self.labelSuffix
            labelPath = self.labelFolderPath + '/' + labelName
            #  [[图片编号,分类,x1,y1,x2,y2]]
            label = BoxProcess.getLabelFromXml(labelPath)
            label = BoxProcess.addPaddingToLable(label,padding,scaleRate)
        except Exception as e:
            print('\033[31m [error label] \033[0m',  'path: ' + labelPath)
            print(e)
            sys.exit()
        return image,label

    def __len__(self):
        return len(self.imageFileNames)

    @staticmethod
    def collect_fn(batch):
        """由于 box 的维度是不统一的，因此要自定义数据集的 batch 整理函数 """
        # 将标签和变量拆解
        images,labels = list(zip(*batch))

        # 为每一个框添加图片标记，之后才直到那个是哪张图片里面的框
        for i,boxs in enumerate(labels):
            boxs[:,0] = i

        # 将 tensor 数组拼接成一个 tensor
        return torch.stack(images,dim=0), torch.cat(labels,dim=0)



if __name__ == '__main__':
    dataset = ImageLabelDataset('../asset/train/')
    
    loader = torch.utils.data.DataLoader(dataset,64,collate_fn=ImageLabelDataset.collect_fn)

    for batch in loader:
        image,label = batch 
        print(label)


