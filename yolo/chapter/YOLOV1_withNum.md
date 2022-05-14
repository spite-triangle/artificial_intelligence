# YOLO - V1

# 1. 网络模型

<p style="text-align:center;"><img src="/artificial_intelligence/image/yolo/yolov1_network.jpg" width="100%" align="middle" /></p>

对于第一版 YOLO 的网络模型就两个部分：<a href="https://spite-triangle.github.io/artificial_intelligence/#/./DeepLearning/README" class="jump_link"> 卷积层、全连接层 </a> 。
- **输入：** 尺寸为 `448x448x3` 的图片，<span style="color:red;font-weight:bold"> 图片尺寸定死 </span>

- **输出：** 图片中被检测目标的位置（矩形框坐标）与被检测物体的分类。

<p style="text-align:center;"><img src="/artificial_intelligence/image/yolo/yolov1_work.webp" width="75%" align="middle" /></p>

# 2. 目标检测原理

<p style="text-align:center;"><img src="/artificial_intelligence/image/yolo/yolov1_idea.webp" width="50%" align="middle" /></p>

1. 将输入图片通过 `7x7` 网格，划分为 `49` 个单元格
2. 每个单元格负责一个检测目标：存储检测目标外接矩形的「中心点坐标」、长宽；存储检测目标的类型。<span style="color:red;font-weight:bold"> 即当检测目标的外接矩形「中心点坐标」位于该单元格内时，就让该单元格全权负责储存这个检测目标的信息。 </span>
3. 每个单元格持有`2`个候选矩形框，会通过置信度选择一个最好的当作预测结果输出
    <p style="text-align:center;"><img src="/artificial_intelligence/image/yolo/yolov1_resultCandidate.png" width="50%" align="middle" /></p>


# 3. 模型输出

V1 版本的输出结果为 `7x7x30` 的一个向量，对该向量进行维度转换得到

<p style="text-align:center;"><img src="/artificial_intelligence/image/yolo/yolov1_result.webp" width="75%" align="middle" /></p>

其中 `7x7` 表示利用 `7x7` 的网格，将输入图片划分为 `49` 个单元格；`30` 表示对每个单元格预测结果的描述：两个目标位置候补框、置信度、目标的分类

<p style="text-align:center;"><img src="/artificial_intelligence/image/yolo/yolov1_resultVector.png" width="75%" align="middle" /></p>

- bounding box 1 ：第一个候补框的参数，外接矩形中心坐标 $(x_1,y_1)$ ；长宽 $(w_1,h_1)$
- confidence 1 ：第一个候补框是待检测目标的置信度
- bounding box 2 ：第二个候补框的参数，外接矩形中心坐标 $(x_1,y_1)$ ；长宽 $(w_1,h_1)$
- confidence 2 ：第二个候补框是待检测目标的置信度
- 分类：检测目标为 `20` 个分类的概率

> [!note]
> 其中，对于中心坐标 $(x,y)$ 、长宽 $(w,h)$ 值的存储是一个百分比。
> - 中心坐标 $(x,y)$ ：相对单元格长宽的比值
> - 长宽 $(w,h)$：相对于输入图片长宽的比值

# 4. 损失函数

## 4.1. 定义

<p style="text-align:center;"><img src="/artificial_intelligence/image/yolo/yolov1_loss.jpg" width="100%" align="middle" /></p>

- $i$：表示对`SxS`的单元格的索引，将二维数组将维为一维数组进行处理
- $j$：对`B`个候补矩形的索引
- $1_i^{obj}$：标记单元格中是否存在检测目标，存在值为`1`，不存在值为`0`。<span style="color:red;font-weight:bold"> 判断模型预测检测目标存不存在，就比较 IOC </span>
- $1_{ij}^{obj}$：标记单元格的「候补框」中是否存在检测目标，存在值为`1`，不存在值为`0`
- $1_{ij}^{noobj}$：标记单元格的「候补框」中是否存在检测目标，存在值为`0`，不存在值为`1`
- $C$：置信度
  - $\hat{C}_i$：是检测目标值为`1`；不是检测目标值为`0`

## 4.2. 位置预测

<p style="text-align:center;"><img src="/artificial_intelligence/image/yolo/yolov1_wh.jpg" width="25%" align="middle" /></p>

当预测外接框与目标外接框的宽度、高度的差值一样时，对于较大的物体而言相对误差小，而对于较小物体而言相对误差较大。因此为了让损失函数对小物体的外接矩形的宽度、高度更敏感一些，在 YOLO V1 中采用了 **「根号」: 自变量在[0,1]取值时，根号的斜率变化比直线要大。**


# 5. 模型预测

## 5.1. 原理

训练好的 YOLO 网络，输入一张图片，将输出一个 `7x7x30` 的张量（tensor）来表示图片中所有网格包含的对象（概率）以及该对象可能的`2`个位置（bounding box）和可信程度（置信度）。为了从中提取出最有可能的那些对象和位置，YOLO采用NMS（Non-maximal suppression，非极大值抑制）算法。

## 5.2. 极大值抑制

**得分：**

$$
Score_{ij} = P_i(C) * C_j
$$

20个对象的概率乘以2个bounding box的置信度，共40个得分（候选对象）。49个网格共1960个得分。Andrew Ng建议每种对象分别进行 NMS，那么每种对象有 `1960/20=98` 个得分。

**NMS**：选择得分最高的作为输出，与该输出重叠的去掉

1. 设置一个Score的阈值，低于该阈值的候选对象排除掉（将该Score设为0）
1. 遍历每一个对象类别，遍历该对象的98个得分
    1. 找到Score最大的那个对象及其bounding box，添加到输出列表
    1. 对每个Score不为0的候选对象，计算其与上面2.1.1输出对象的bounding box的IOU
    1. 根据预先设置的IOU阈值，所有高于该阈值（重叠度较高）的候选对象排除掉（将Score设为0）
    1. 如果所有bounding box要么在输出列表中，要么Score=0，则该对象类别的NMS完成，返回步骤2处理下一种对象
1. 输出列表即为预测的对象

