# YOLO - SPP

# 1. SPP介绍

YOLO SPP 属于是 YOLO V3 的 PRO 改进版本。在原始 yolo v3 的基础上增加了 Mosaic 数据增强、spp网络层、位置损失函数改进。

# 2. Mosaic 图像增强

- **思路：** 将四张图片进行随机裁剪，再拼接到一张图上作为训练数据

    <p style="text-align:center;"><img src="/artificial_intelligence/image/yolo/mosaic.jpg" width="75%" align="middle" /></p>

- **优点：**
  1. 增加样本多样性
  2. 增加一张图片中的目标数，利于多目标训练
  3. Batch Normal 一次能统计多张图片的结果

# 3. SPP 网络

<p style="text-align:center;"><img src="/artificial_intelligence/image/yolo/yolo_spp.jpg" width="75%" align="middle" /></p>

YOLO SPP 的网络在原v3的基础上增加了一层 SPP 网络层。SPP 网络的工作原理是对输入进行三次不同 kernel 的最大池化操作，且**池化后特征图的尺寸不变**，最后的三次池化结果在通道维度上直接拼接作为输出结果。

<p style="text-align:center;"><img src="/artificial_intelligence/image/yolo/sppNet.jpg" width="50%" align="middle" /></p>

# 4. IOU 位置损失

## 4.1. L2 位置损失缺陷

<p style="text-align:center;"><img src="/artificial_intelligence/image/yolo/yolospp_l2.jpg" width="50%" align="middle" /></p>

上面三种预测框与 Ground True Box 的匹配情况，原 v3 的 L2 位置损失函数计算得到损失结果是一样的，但是根据肉眼观察，明显第三种预测要优于其余两种预测结果。**因此，L2 损失不能很好反应预测框与真实框相重合的优劣程度。**

## 4.2. IOU 损失

- **原理：** 基于预测框与 Ground True Box 的 IOU 来反映预测结果与期望结果的位置损失

    $$
    IoU \ loss = 1 - \rm IoU
    $$

- **缺点：IOU 的取值范围为 [0,1]。当预测框与 Ground True Box 不存在重叠部分时，IOU恒等于0。这就导致预测框与 Ground True Box 不管相差多远，损失值都是恒定的，不利用网络学习。** 

## 4.3. GIOU 损失

- **GIOU定义：**

    $$
    GIoU = IoU - \frac{A^c - u}{A^c}
    $$

    <p style="text-align:center;"><img src="/artificial_intelligence/image/yolo/ciou.jpg" width="25%" align="middle" />

    其中 $A^c$ 为预测框与 Ground True Box的外接矩形框面积（蓝色线框）；$u$ 为预测框与 Ground True Box的并集。当预测框与 Ground True Box完全重合时，$GIoU = 1$；当预测框与 Ground True Box 距离无限远时，$GIoU = -1$

- **GIOU损失：**
  $$
  GIoU \ loss = 1 - GIoU 
  $$

- **GIOU缺点：** 当预测框与 Ground True Box 高度（宽度）一样时，在水平（垂直）方向上，$GIoU$ 会退化为 $IoU$

  <p style="text-align:center;"><img src="/artificial_intelligence/image/yolo/GIOU2IOU.jpg" width="25%" align="middle" /></p>

## 4.4. DIOU 损失

- **DIOU定义：**
  $$
  DIoU = IoU - \frac{d^2}{c^2}
  $$
  <p style="text-align:center;"><img src="/artificial_intelligence/image/yolo/DIOU.jpg" width="25%" align="middle" /></p>

  其中 $d$ 为预测框与 Ground True Box中心点之间的距离

- **DIOU 损失：**
  $$
  DIoU \ loss = 1 - DIoU
  $$

- **优点：** 收敛速度要快于 IOU 与 GIOU 损失

## 4.5. CIOU 损失

- **CIOU定义：**
  $$
  \begin{aligned}
    CIoU &= IoU - \frac{d^2}{c^2} - \alpha v \\
    v &= \frac{4}{\pi^2} (\arctan \frac{w^{gt}}{h^{gt}} - \arctan \frac{w}{h})^2 \\
    \alpha &= \frac{v}{1 - IoU + v}
  \end{aligned}
  $$

- **CIOU损失：**
  $$
  CIoU \ loss = 1 - CIoU 
  $$

- **优点：** DIOU 只考虑了预测框与 Ground True Box 重叠面积、中心距离的几何差异，而 CIOU 在此基础上，增加了「长宽比」的损失，使得损失的计算精度更高。

