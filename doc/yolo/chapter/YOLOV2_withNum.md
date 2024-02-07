# YOLO - v2

# 1. 模型改进

<p style="text-align:center;"><img src="/artificial_intelligence/image/yolo/darknet19.png" width="75%" align="middle" /></p>

## 1.1. 卷积化

在 V1 中，最后的输出结果是靠「全连接层」得到的，这也就限制了输入图片的尺寸。因此在 V2 将所有的全连接层转为了卷积层，构造了新的网络结果 DarkNet19，其中还利用 1x1 卷积对模型进行优化。

- <a href="https://spite-triangle.github.io/artificial_intelligence/#/./DeepLearning/chapter/convolution" class="jump_link"> 全连接层卷积化 </a>

- <a href="https://spite-triangle.github.io/artificial_intelligence/#/./DeepLearning/chapter/convolution" class="jump_link"> 1x1 卷积 </a>

## 1.2. Batch Nomalization

在 DarkNet19 网络中，对于卷积层加入了 <a href="https://spite-triangle.github.io/artificial_intelligence/#/./DeepLearning/chapter/improveNetwork" class="jump_link"> Batch Normalization </a> ，并删除了 <a href="https://spite-triangle.github.io/artificial_intelligence/#/./DeepLearning/chapter/improveNetwork" class="jump_link"> dropout </a> 。



由于 DarkNet19 做了 `5` 次池化且卷积均进行了padding，所以输入图片将会被缩放 $2^5=32$ 倍，即 `448x448` 的输入，输出结果应当是 `448/32=14`，但是`14x14`的结果没有特定的中心点，为了制造一个中心点，模型的输入图片尺寸就更改为了`416x416`，输出结果就变为了`13x13`。

## 1.3. Fine-Grained Features

<p style="text-align:center;"><img src="/artificial_intelligence/image/yolo/yolov1_fineGrain.png" width="50%" align="middle" /></p>

在 DarkNet19 模型中，存在一个 PassThrough Layer 的操作，该操作就是将之前阶段的卷积层结果与模型输出结果进行相加。根据 <a href="https://spite-triangle.github.io/artificial_intelligence/#/./DeepLearning/chapter/convolution" class="jump_link"> 感受野 </a> 可知，越靠前的网络层对细节的把握越好，越靠后的网络更注重于目标整体，为了使得输出结果对小物体有更好的把握，就可以利用 PassThrough Layer 来提升结果特征图对小物体的敏感度。

## 1.4. 图片输入

由于历史原因，ImageNet分类模型基本采用大小为 `224x224` 的图片作为输入，所以 YOLO V1 模型训练使用的输入图片大小其实为 `224x224`，在模型预测时，又使用的是 `448x448` 的图片作为输入，这样就导致模型的训练和模型的预测，输入其实是有差异的。为了弥补这个差异，模型训练的最后几个 epoch 采用 `448x448` 的图片进行训练。

# 2. Anchor Box

**YOLO V1 的 bounding box 缺陷**：
  1. 一个单元格只能负责一个目标检测的结果，如果该单元格是多个目标的中心点区域时，V1 版本将不能识别。
  2. V1 中，对于 bounding box 的预测结果并未加限制，这就会导致 bounding box 的中心点可能会跑到其他单元格内
  3. bounging box 的宽度与高度是靠模型自己学习的，这就可能走很多弯路。

## 2.1. Anchor

<p style="text-align:center;"><img src="/artificial_intelligence/image/yolo/yolov2_anchor.png" width="50%" align="middle" /></p>

1. 模型训练开始前，人为为每个单元格预定义几个不同大小的 Anchor Box，这样从训练开始，每个单元格的bounding box就有了各自预先的检测目标，例如瘦长的 bounging box 就适合找人，矮胖的 bounging box 就适合找车等。

2. 模型训练就是调整这些预定义的bounding box 的中心点位置与长宽比列。

## 2.2. Box 数据结构

<p style="text-align:center;"><img src="/artificial_intelligence/image/yolo/yolov2_anchorOutput.png" width="50%" align="middle" /></p>

<p style="text-align:center;"><img src="/artificial_intelligence/image/yolo/yolov2_output.png" width="75%" align="middle" /></p>


在引入 Anchor box 后， YOLO V2 对于一个 bounging box 的数据结构为：
- 中心点坐标 $(x,y)$，相对于单元格宽度的比列值
- 相对于 Anchor box 宽高的偏移量 $(w,h)$
- 当前 bounging box 存在检测目标的置信度 $Confidence$ 
- 检测目标对应各个类型的概率 $p_i$

一个单元格能检测多少目标，就输出多少个上述 Box 数据结构。

## 2.3. Box 解析

<p style="text-align:center;"><img src="/artificial_intelligence/image/yolo/yolov2_boundingbox.png" width="50%" align="middle" /></p>

假设一个 bounding box 的输出结果为 

$$
[t_x,t_y,t_h,t_w,t_o]
$$


-  **bounding box 的中心点坐标**

    $$
    \begin{aligned}
        b_x = \sigma(t_x) + C_x \\
        b_y = \sigma(t_y) + C_y \\
    \end{aligned}
    $$

    $(C_x,C_y)$ 为单元格在输出网格中，左上角的坐标；<span style="color:red;font-weight:bold"> $\sigma()$ 为 sigmod 函数，为了将中心点限制在当前单元格内 </span>。

    <p style="text-align:center;"><img src="/artificial_intelligence/image/yolo/sigmod.png" width="50%" align="middle" /></p>

- **bounding box 的宽度**

    $$
    \begin{aligned}
        b_w = P_w e^{t_w} \\
        b_h = P_h e^{t_h}
    \end{aligned}
    $$

    $( P_w,P_h )$ 为该 bounding box 对应的 Anchor box 的大小；$e^{()}$ 的目的和 YOLO V1 一样，只是将原来的 $\sqrt{()}$ 替换成了 $\ln()$。

- **bounding box 的置信度**
    $$
    Confidence = \sigma(t_o)
    $$

## 2.4. 确定 Anchor 

- **K类聚：**
  1. 收集所有训练样本图片中检测目标的期望 bounding box，<span style="color:red;font-weight:bold"> 只要宽度、高度；不要中心点坐标 </span>
  2. 利用 K-means 类聚算法将上述样本框划分为 k 类，距离计算公式为
      $$
      d_{center} = 1 - IOU(center,box)
      $$
  3. 选择每个类别的中心 bounding box 作为 Anchor box ，一共 k 个

- **k 的确定：** 作者对 VOC 与 COCO 数据集，进行测试后，选择了 $k = 5$

    <p style="text-align:center;"><img src="/artificial_intelligence/image/yolo/yolov2_kAnchor.png" width="50%" align="middle" /></p>

# 3. 损失函数

<p style="text-align:center;"><img src="/artificial_intelligence/image/yolo/yolov2_loss.png" width="75%" align="middle" /></p>

