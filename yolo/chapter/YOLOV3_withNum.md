# YOLO - V3

# 1. 网络模型

<p style="text-align:center;"><img src="/artificial_intelligence/image/yolo/yolov3_model.png" width="75%" align="middle" /></p>

YOLOV3 相对于 YOLOV2 而言，着重对网络结构进行了改进。
- 在网络中引入的 <a href="https://spite-triangle.github.io/artificial_intelligence/#/./DeepLearning/chapter/convolution" class="jump_link"> 残差网络 </a>
- **没有池化层**，使用步幅为2的卷积层替代池化层进行特征图的降采样过程
- **多尺度预测结果**：根据 <a href="https://spite-triangle.github.io/artificial_intelligence/#/./DeepLearning/chapter/convolution" class="jump_link"> 感受野 </a> ，将网络主干下层的卷积结果与上层的卷积结果进行合并，使得上层卷积结果对「物体局部」敏感的同时也能兼顾「物体整体」的信息。因此，利用最上层的结果预测小物体，利用下层的卷积结果预测大物体。


<p style="text-align:center;"><img src="/artificial_intelligence/image/yolo/darknet53.png" width="75%" align="middle" /></p>

<details>
<summary><span class="details-title">主干网络参数</span></summary>
<div class="details-content"> 

<p style="text-align:center;"><img src="/artificial_intelligence/image/yolo/yolo3_darknet53.webp" width="75%" align="middle" /></p>


</div>
</details>

# 2. Anchor Box

对于三类预测结果而言，不同尺度的预测结果具有不同的「感受野」

<p style="text-align:center;"><img src="/artificial_intelligence/image/yolo/yolov3_prespective.png" width="75%" align="middle" /></p>

为每一个预测结果分别配置三类 Anchor Box。对于感受野较大的预测结果分配较大的三类 Anchor Box；对于感受野较小的预测结果分配较小的三类 Anchor Box。


<p style="text-align:center;"><img src="/artificial_intelligence/image/yolo/yolov3_anchorBox.png" width="75%" align="middle" /></p>

# 3. 预测结果

<p style="text-align:center;"><img src="/artificial_intelligence/image/yolo/yolov3_scales.png" width="75%" align="middle" /></p>

输入图片根据尺度的不同，输出结果为具有三中网格尺寸的特征结果。每个网格中的单元格均对应了三个 Anchor Box 长宽、位置的修正值 $(t_x,t_y,t_w,t_h)$，以及 Anchor Box 中检测目标是否存在的概率 $P_o$ 、分类概率 $P_i$。

<p style="text-align:center;"><img src="/artificial_intelligence/image/yolo/yolov3_cell.png" width="50%" align="middle" /></p>


**分类概率与检测目标存在概率相乘，就能得到该 Anchor Box 中的检测目标种类概率**

<p style="text-align:center;"><img src="/artificial_intelligence/image/yolo/yolov3_scores.png" width="75%" align="middle" /></p>

# 4. 损失函数

<p style="text-align:center;"><img src="/artificial_intelligence/image/yolo/yolov3_selectAnchor.png" width="50%" align="middle" /></p>

- **预测框种类划分**：假设 Ground True Box 与 Anchor Box 中心重合，然后计算 IOU。（**这里就能实现，对预先给定的 Anchor Box 进行筛选**）
  - **正样本：** Ground True Box 对应位置；IOU 最大。虚线框
  - **舍弃的样本：** Ground True Box 对应位置；IOU 并非最大，但是又大于指定阈值。点虚线框
  - **负样本：** 所在单元格没有 Ground True Box。黄色框


- **损失函数**
    <p style="text-align:center;"><img src="/artificial_intelligence/image/yolo/yolov3_loss.png" width="100%" align="middle" /></p>

> [!note|style:flat]
> - 在损失函数的实际实现时，置信度使用的时「BCE 损失函数」，分类使用的是「CE 损失函数」
> - **别被上面的公式所误导，正负样本的置信度损失要一起计算，即放到同一个「BCE函数」进行计算，若分开实现的话，训练效果很扯淡 (￣_￣|||)**