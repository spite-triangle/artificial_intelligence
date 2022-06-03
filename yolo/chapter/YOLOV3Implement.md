# YOLO V3 代码实现

# 项目简介

- **工程目的：** 纯手撸 YOLO v3， 实现对口罩的检测。
    <p style="text-align:center;"><img src="../../image/yolo/mask_no.jpg" width="50%" align="middle" /></p>

    <p style="text-align:center;"><img src="../../image/yolo/mask_have.jpg" width="50%" align="middle" /></p>

- <a href="https://github.com/spite-triangle/artificial_intelligence/tree/master/example/detectMask" class="jump_link"> 项目工程 </a>

- **训练素材**：来自 <a href="https://www.bilibili.com/video/BV1c64y1e7wY" class="jump_link">B站 UP炮哥带你学 </a>

> **注意：** 
> - 由于 GitHub 上的 <a href="https://github.com/eriklindernoren/PyTorch-YOLOv3" class="jump_link"> yolo v3  实现</a> 偏向实际应用，其代码实现不利于学习，因此，本项目着重于对 yolo v3 原理的复现，一些花里胡哨的功能就全部忽略掉了（例如，原项目中的`.cfg`配置文件）。
> - 本项目中给出的模型权重参数（在 `weightsBackup` 文件夹里），由于训练次数少、训练素材少，模型复现简化等原因，其精度并不是很好，但是能玩 ( *︾▽︾)。
> - 由于本人是外行，能力有限，若发现存在偏差，不用怀疑，那肯定是我的问题，请多多包涵 ( •̀ ω •́ )✧


# 项目运行
- **环境需求：**
    > - Python
    > - PyTorch
    > - OpenCV
- **运行：** 下载项目工程后，直接运行对应的 .py 文件即可。默认是启动了`GPU`，如有需要，请在 `config` 中修改相关配置

# 目录结构

```term
triangle@LEARN:~$ tree  detectMask/
.
├── ManageData/
│   └── Datasets.py
├── Model/
│   ├── Loss.py
│   └── Network.py
├── Utils/
│   ├── BoxProcess.py
│   ├── ImageProcess.py
│   └── PostProcess.py
├── asset/
│   ├── images/
│   ├── testSet/
│   │   ├── test_images/
│   │   └── test_labels/
│   ├── trainSet/
│   │   ├── train_images/
│   │   └── train_labels/
│   ├── videos/
│   ├── weights/
│   └── weightsBackup/
└── config/
├── detectImage.py
├── detectVideo.py
└── train.py
```

- `ManageData`：训练集与测试集的数据管理模块
- `Model`：Darknet 53 模型的实现；yolo v3 损失函数的实现
- `Utils`：工具模块
  - **BoxProcess**：外接矩形相关处理函数
  - **ImageProcess**：图片展示、变换、绘制文字与外接矩形等相关处理函数
  - **PostProcess**：极大值抑制得到最终预测结果；模型评价指标计算 precision , recall , ap
- `asset`：存放模型测试与训练相关的视频、图片、模型权重文件资源。
  - **images 与  videos**：用来测试模型的图片与视频
  - **trainSet**：模型训练的训练集
  - **testSet**：模型训练的测试集
  - **weights**：模型训练过程中，保存的权重参数
  - **weightsBackup**：备份的权重参数
- `config`：模型的配置参数
- **train.py**：训练模型
- **detectImage.py**：利用图片进行模型测试
- **detectVideo.py**：利用视频进行模型测试



