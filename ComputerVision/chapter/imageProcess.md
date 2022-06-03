# 图像处理

# 图像金字塔

<p style="text-align:center;"><img src="../../image/computerVision/ImagePyramid.jpg" width="50%" align="middle" /></p>

- **作用：** 将图像根据分辨率大小分为多个档次。

## 高斯金字塔

- 向下采样（缩小图片）：
  1. 首先进行高斯滤波
  2. 去除偶数的行、列

- 向上采用（放大图片）：
    1. 用「零」填充偶数行、列
        <p style="text-align:center;"><img src="../../image/computerVision/GaussPyramid.jpg" width="50%" align="middle" /></p>
    2. 对放大的图片进行高斯卷积，将「零」值进行填充

```python
# 向上采样
cv2.pyrUp(src[, dst[, dstsize[, borderType]]]) -> dst

# 向下采样
cv2.pyrDown(src[, dst[, dstsize[, borderType]]]) -> dst
```

## 拉普拉斯金字塔

$$
I_{i+1} = I_{i} - \rm PyrUp(pyrDown(I_{i}))
$$

迭代执行上面的公式，就能得到每一层的图像。

# 图像轮廓

> [!note]
> 轮廓定义：构成任何一个形状的边界或外形线，是指将「边缘」连接起来形成的一个整体。

## 轮廓提取

```python

# contours：从图像中查找出来的轮廓数组
# hierarchy：轮廓层级
# imageSrc：传入的图像，又返回了一份。不明白。。。。
cv2.findContours(image, mode, method[, contours[, hierarchy[, offset]]]) ->imageSrc, contours, hierarchy
```

- mode: 轮廓检索模式
  - RETR_EXTERNAL:只检索最外面的轮廓；
  - RETR_LIST:检索所有的轮廓，并将其保存到一条链表当中；
  - RETR_CCOMP:检索所有的轮廓，并将他们组织为两层：顶层是各部分的外部边界，第二层是空洞的边界：
  - **RETR_TREE**: 检索所有的轮廓，并重构嵌套轮廓的整个层次；<span style="color:red;font-weight:bold"> 最常用。 </span>
- method: 重新绘制轮廓的算法
  - CHAIN_APPROX_NONE:以Freeman链码的方式输出轮廓，轮廓信息完整保留
  - CHAIN_APPROX_SIMPLE:压缩水平的、垂直的和斜的部分，只保留顶点。
  <p style="text-align:center;"><img src="../../image/computerVision/approxContoursMethod.jpg" width="75%" align="middle" /></p>


> [!note|style:flat]
> 用于轮廓检测的图像，首先得进行二值处理（阈值操作）或 Canny 边缘检测。

## 轮廓绘制

```python
# canvas：轮廓要绘制在哪张背景图上，直接覆盖原图
# contours：findContours 找到的轮廓信息
# contourIdx：轮廓数组contours的索引值，-1 为全部
# color：轮廓颜色
# thickness：轮廓厚度
cv2.drawContours(canva:image, contours, contourIdx, color[, thickness[, lineType[, hierarchy[, maxLevel[, offset]]]]]) -> image
```

<details>
<summary><span class="details-title">完整代码</span></summary>
<div class="details-content"> 


```python
import cv2
import numpy as np
# 读取图片
img = cv2.imread('./morphology.jpg')
# 灰度图
imgGray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
# 二值化
retval,imgBinary = cv2.threshold(imgGray,127,255,cv2.THRESH_BINARY)

# 提取轮廓
imageCrc,contours, hierarchy=cv2.findContours(imgBinary,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)

# 绘制轮廓
canva = img.copy()
imgRes = cv2.drawContours(canva,contours,-1,(0,0,255),1)

cv2.imshow('contours',imgRes)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

<p style="text-align:center;"><img src="../../image/computerVision/shapecontours.jpg" width="50%" align="middle" /></p>

</div>
</details>

## 轮廓特征

```python
# 轮廓索引
cnt = contours[0]

# 计算面积
area = cv2.contourArea(cnt)

# 计算周长
# arcLength(curve, closed) -> retval
arc = cv2.arcLength(cnt,True)
```

## 轮廓近似

<p style="text-align:center;"><img src="../../image/computerVision/approxContour.jpg" width="50%" align="middle" /></p>

近似弧线 $\stackrel\frown{AB}$，首先连接A、B两点做直线 $\overline{AB}$；然后找 $\stackrel\frown{AB}$ 到 $\overline{AB}$ 最长的距离，假设$C$距离$\overline{AB}$最大，且距离为 $d$；最后对比 $d$ 与阈值 $\epsilon$ 的大小，若 $d < \epsilon$，则用直线 $\overline{AB}$ 近似曲线 $\stackrel\frown{AB}$，否则将$\stackrel\frown{AB}$ 拆分为 $\stackrel\frown{AC}$ 与 $\stackrel\frown{CB}$ 重复上述步骤。

```python
# curve：轮廓，contour
# epsilon：阈值，按照周长百分比选取 arcLength
# closed：近似轮廓是否闭合
cv2.approxPolyDP(curve, epsilon, closed[, approxCurve]) -> approxCurve
```

## 轮廓标记

**作用：** 用一个形状（矩形、圆圈等）将轮廓标记出来。

<!-- panels:start -->
<!-- div:left-panel -->

```python
# 背景画布
canvabg = img.copy()

# 获取轮廓
cnt0 = contours[0]
# 矩形边框
startx,starty,width,height = cv2.boundingRect(cnt0)
cv2.rectangle(canvabg,(startx,starty),(startx + width,starty + height),(0,255,0),2)

# 获取轮廓
cnt1 = contours[2]
# 圆圈外框
(cx,cy),radius = cv2.minEnclosingCircle(cnt1)
cv2.circle(canvabg,(int(cx),int(cy)),int(radius),(255,0,0),2)
```
<!-- div:right-panel -->
<p style="text-align:center;"><img src="../../image/computerVision/outline.jpg" width="50%" align="middle" /></p>
<!-- panels:end -->

# 模板匹配

- **思路：** 将模板图片当作卷积核与被匹配的图片进行卷积操作，然后根据具体 <a href="https://docs.opencv.org/4.0.1/df/dfb/group__imgproc__object.html#ga3a7850640f1fe1f58fe91a2d7583695d" class="jump_link"> 匹配算法 </a> 计算出每一步卷积操作的置信度，根据置信度来确定模板图像在被匹配图像中的位置。

```python
# templ：模板图片
# method：匹配算法
cv2.matchTemplate(image, templ, method[, result[, mask]]) -> result
```
- method:
  - TM_SQDIFF:计算平方不同，计算出来的值越小，越相关
  - TM_CCORR:计算相关性，计算出来的值越大，越相关
  - TM_CCOEFF:计算相关系数，计算出来的值越大，越相关
  - TM_SQDIFF NORMED:计算归一化平方不同，计算出来的值越接近0，越相关
  - TM_CCORR NORMED:计算归一化相关性，计算出来的值越接近1，越相关
  - TM_CCOEFF NORMED:计算归一化相关系数，计算出来的值越接近1，越相关

- result：每一步卷积操作记录一次结果，其数组大小就为（与卷积运算结果维度计算一样）
  $$
  \begin{aligned}
    width = W_{src} - W_{temp} + 1 \\
    height = H_{src} - H_{temp} + 1 \\
  \end{aligned}
  $$
  **result数组的索引值，对应的是模板图片在原始图片重合的左上角像素的坐标。**

<!-- panels:start -->
<!-- div:left-panel -->
```python
# 导入图片
img = cv2.imread('./cat.jpeg')
imgTemp = img[80:250,250:440]

# 模板匹配
result = cv2.matchTemplate(img,imgTemp,cv2.TM_SQDIFF_NORMED)

# 统计出数组的中最大值、最小值以及对应的索引
minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(result)

# 绘制矩形框
cv2.rectangle(img,minLoc,(minLoc[ 0 ]+imgTemp.shape[ 1 ],minLoc[ 1 ]+imgTemp.shape[ 0 ]),(255,0,0),2)
```
<!-- div:right-panel -->
<p style="text-align:center;"><img src="../../image/computerVision/matchTemplate.jpg" width="50%" align="middle" /></p>
<!-- panels:end -->

- **一个模板匹配多个：** 遍历匹配结果数组，找到所有置信度满足要求的像素坐标点。

# 直方图

## 对比度

- **定义：** 一幅图像中明暗区域最亮的白和最暗的黑之间不同亮度层级的测量，即指一幅图像灰度反差的大小。差异范围越大代表对比越大，差异范围越小代表对比越小。<span style="color:red;font-weight:bold"> 说人话，应该就是图片灰度图的明暗分布明显，数学上就是灰度值的差异大。 </span>

- **图像对比度：**
  $$
  C = \sum_\delta \delta(i,j)^2 P_\delta(i,j)
  $$

  其中 $\delta(i,j)$ 表示 $(i,j)$ 位置灰度值与「邻近」灰度值的差；$P_\delta(i,j)$ 表示 $\delta(i,j)$ 占总 $\delta()$ 的概率分布。表示「邻近」有以下两种方式（`1`表示邻近位置），左边为「四近邻」，右边为「八近邻」

  $$
  \begin{bmatrix}
    0 & 1 & 0 \\
    1 & (i,j) & 1 \\
    0 & 1 & 0 \\
  \end{bmatrix} \qquad
  \begin{bmatrix}
    1 & 1 & 1\\
    1 & (i,j) & 1\\
    1 & 1 & 1\\
  \end{bmatrix}
  $$

  <details>
  <summary><span class="details-title">对比度计算案例</span></summary>
  <div class="details-content"> 
  
  <p style="text-align:center;"><img src="../../image/computerVision/ContrastExample.png" width="50%" align="middle" /></p>

  - 采用「四近邻」法
  - 第一个 $(1^2 + 2^2)$ 表示第一行一列灰度值与邻近的灰度值差的平方：$(1-2)^2 + (1-3)^2$。其余同理。
    <p style="text-align:center;"><img src="../../image/computerVision/contrastDelta.jpg" width="25%" align="middle" /></p>
  - $\frac{1}{48}$ ：计算公式已经将所有的 $\delta(i,j)$ 都给列举出来了， 那么计算 $P_\delta(i,j)$ 只要再除以 $\delta()$ 的总个数就行了。
    
  </div>
  </details>




## 绘制直方图

<p style="text-align:center;"><img src="../../image/computerVision/imageHistogram.jpg" width="50%" align="middle" /></p>

直方图的横坐标为像素通道值的取值范围；纵坐标为数值出现的次数。

```python
# OpenCV 方法
# images：图像，输入 [ image ]
# channels：选择通道，输入 [ channel ]
# mask：遮罩
# hisSize：有几根柱子，输入 [ hisSize ]
# range：取值范围
cv2.calcHist(images: List[Mat], channels: List[int],
       mask: Mat | None, histSize: List[int], ranges: List[int]) -> hist

# matplotlib 方法
# data ：要绘制直方图的一维数据
# hisSize：柱子的个数 
plt.hist(data,hisSize)
```

> [!tip] 
> 推荐使用 matplotlib 方式，OpenCV 方式最后还得用 matplotlib 进行绘图。

## 均衡化

### 理论
<p style="text-align:center;"><img src="../../image/computerVision/idealEqualization.jpg" width="50%" align="middle" /></p>

- **目的：** 将原图像通过变换，得到一幅灰度直方图的「灰度值均匀分布」的新图像。对在图像中像素个数多的灰度级进行展宽，而对像素个数少的灰度级进行缩减。从而达到清晰图像的目的。**最理想的情况就是变换后，像素灰度概率是完全一样的，但是实际上做不到那么平均。**
<p style="text-align:center;"><img src="../../image/computerVision/equalization.jpg" width="75%" align="middle" /></p>

- **算法流程：** 首先统计出灰度值与其出现次数的直方图；然后对灰度值升序排序；接着计算出现概率（出现次数 / 总像素），并根据灰度值从低到高计算累计概率（当前概率 + 之前的总概率）；最后根据公式：累计概率 * （位深最大值 - 0），将数值映射到[位深最大值,0]。

- **均衡化：** <span style="color:red;font-weight:bold"> 直接假设输出灰度的概率就是均匀的 $p=\frac{1}{w \times h}$，然后才推导转换公式。但是一顿操作下来，只修改了图像灰度值，并未对灰度概率进行修改（灰度概率改成均匀的，图像不就被彻底修改了）。所以算法从结果上来看，是实现了图片所涉及的灰度值分布更均匀一些，而非直方图灰度概率分布。</span>

- <a href="https://blog.csdn.net/j05073094/article/details/120251878" class="jump_link"> 公式推导 </a>


### OpenCV 代码

```python
cv2.equalizeHist(src:image[, dst]) -> dst:image
```

<details>
<summary><span class="details-title">案例代码</span></summary>
<div class="details-content"> 

```python
import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('./cat.jpeg')

# 转换颜色空间：主要为了对 灰度 进行均值化
yuv =cv2.cvtColor(img,cv2.COLOR_BGR2YUV)

# 均衡化
yEqul = cv2.equalizeHist(yuv[:,:,0])

# 替换原来的灰度
yuv[:,:,0] = yEqul

# 还原颜色空间
imgEual = cv2.cvtColor(yuv,cv2.COLOR_YUV2BGR)

cv2.imshow('match',np.hstack((img,imgEual)))
cv2.waitKey(0)
cv2.destroyAllWindows()
``` 

</div>
</details>

<p style="text-align:center;"><img src="../../image/computerVision/equalizationImage.jpg" width="50%" align="middle" /></p>

## CLAHE

### 理论

- **直方图均衡化问题：** 
  - 为全局效果，这就导致图像中原来暗部和亮部的细节丢失，例如上图猫的帽子和左脚处。
  - 可能导致噪点的放大。

- **思路：** 将图片拆分为多个部分，然后每个部分分别进行均衡化处理，且对每个部分的直方图概率分布做限制（防止某个灰度值的概率分布过大，进而导致均衡化后的灰度值过大）。


- **算法实现：**
  1. 图像分块
      <p style="text-align:center;"><img src="../../image/computerVision/claheBlock.jpg" width="25%" align="middle" /></p>

  2. 找每个块的中心点（黄色标记）
      <p style="text-align:center;"><img src="../../image/computerVision/claheBlockCenter.jpg" width="25%" align="middle" /></p>

  3. 分别计算每个块的灰度直方图，并进行「阈值限制」
  
      <p style="text-align:center;"><img src="../../image/computerVision/claheHistogram.jpg" width="50%" align="middle" /></p>

      绘制好直方图后，柱子的分布值与设定「阈值」进行比较，超过阈值的部分则进行裁剪，并均匀分配给所有的柱子。分配后，直方图又要柱子超出时（绿色部分），继续重复上述操作，直至直方图柱子都在「阈值」下方。<span style="color:red;font-weight:bold"> 现在只是对「直方图分布」进行修改，并没有修改原始图像的任何内容。 </span>
    
  4. 得到每个块的直方图分布后，**根据直方图均衡化算法对每个块的中心点（黄色标记）进行均衡化处理**。<span style="color:red;font-weight:bold"> 只对中心点进行均衡化是为了加快计算速度，对每一个像素都进行处理会浪费很多时间。 </span>

  5. 根据中心点均衡化后的灰度值，利用插值算法计算图像块剩余像素的灰度值。**插值算法计算效果和直接均衡化效果差不多，但是差值计算速度更快。**

### OpenCV 代码

```python
# 生成自适应均衡化算法 
# clipLimit ：阈值，1 表示不做限制。值越大，对比度越大
# tileGridSize：如何拆分图像
clahe = cv2.createCLAHE([, clipLimit[, tileGridSize]]) -> retval

# 对像素通道进行自适应均值化处理
dst = clahe.apply(src)
```

<p style="text-align:center;"><img src="../../image/computerVision/clahe.jpg" width="50%" align="middle" /></p>


# 图像傅里叶变换

> -  <a href="https://zhuanlan.zhihu.com/p/19763358" class="jump_link"> 傅里叶变换掐死教程（说人话版） </a>
> - <a href="https://spite-triangle.github.io/algorithms/fastFourier/Fourier.html" class="jump_link"> 一维傅里叶变换（数学精简版）</a>
> - <a href="https://spite-triangle.github.io/algorithms/digitalSignalProcessing/digitalSignalProcessing.html" class="jump_link"> 数字信号处理（一维傅里叶完整版） </a>
> - <a href="https://zhuanlan.zhihu.com/p/110026009" class="jump_link"> 二维傅里叶变换（说人话版） </a>
> - <a href="https://zhuanlan.zhihu.com/p/99605178" class="jump_link"> 图像傅里叶（说人话版） </a>

## 二维傅里叶变换

- **思想：** 二维傅里叶变换中，认为二维数据是由无数个「正弦平面波」所构成。

<p style="text-align:center;"><img src="../../image/computerVision/fourier2d.jpg" width="75%" align="middle" /></p>

- **离散傅里叶变换公式：**

  $$
  F(u, v)=\sum_{x=0}^{M-1} \sum_{y=0}^{N-1} f(x, y) e^{-j 2 \pi\left(\frac{\mathrm{ux}}{\mathrm{M}}+\frac{v y}{N}\right)}
  $$

  **将二维数据进行傅里叶变换后得到的值 $F(u,u)$ 则代表了相应的「正弦平面波」**

## 正弦平面波

<p style="text-align:center;"><img src="../../image/computerVision/sinPlane.jpg" width="50%" align="middle" /></p>

- **直观定义：** 将一维正弦曲线朝着纵向的一个方向上将其拉伸得到一个三维的波形，然后将波形的幅值变化用二维平面进行表示，再将二维平面波绘制成灰度图，即波峰为白色、波谷为黑色。

- **数学参数：**  
  - 正弦波：频率 $w$ ，幅值 $A$ ，相位 $\varphi$
  - 拉伸方向：在二维坐标中，向量可以写为 $\vec{n} = (u,v)$

## 二维傅里叶变换结果 $F(u,v)$ 

- $(u,v)$：拉伸方向的向量
- $w=\sqrt{u^2 + v^2}$：$(u,v)$向量的模表示正弦波频率
- $F(u,v)$：复数，隐含了正弦波的幅值 $A$ 和相位 $\varphi$。下面用一维做解释，二维太复杂也不直观（主要是太难了，不想推。。。。）
  
  $$
  \begin{aligned}
    只考虑这一个变换：&F(x) = a + ib，且 A = \sqrt{a^2 + b^2}，\varphi =\arctan \frac{b}{a} \\
    傅里叶逆变换：&f(x) = F(x) e^{iwx} \\
    & \quad \quad = (a+ib) e^{iwx} \\
    & \quad \quad = A (\frac{a}{A} + i \frac{b}{A}) e^{iwx} \\
    & \quad \quad = A (\cos(\varphi) + i \sin(\varphi)) e^{iwx} \\
   根据欧拉公式 :& \quad \quad = A e^{i\varphi} e^{iwx} \\
   & \quad \quad = A e^{i(wx + \varphi)} \\
  \end{aligned}
  $$

  **$A$ 就是幅值；$\varphi$ 就是相位。**


## 傅里叶变换实现


- **傅里叶变换**

  ```python
  # 图片读取
  img = cv2.imread('./cat.jpeg')
  yuv =cv2.cvtColor(img,cv2.COLOR_BGR2YUV)

  # 将灰度值转浮点类型
  yfloat = np.float32(yuv[:,:,0])

  # 傅里叶变换
  # src：浮点类型数组
  # flags：cv2.DFT_
  # dft(src:np.float[, dst[, flags[, nonzeroRows]]]) -> dst
  dft = cv2.dft(yfloat,flags=cv2.DFT_COMPLEX_OUTPUT)

  # 计算模，也就是幅值
  A = cv2.magnitude(dft[:,:,0],dft[:,:,1])

  # 幅值太大了，重新映射到 (0 - 255)，方便显示
  A = A / A.max() * 255

  cv2.imshow('gray',yuv[:,:,0])
  cv2.imshow('dft',A)
  ```

  <p style="text-align:center;"><img src="../../image/computerVision/dftResult.jpg" width="50%" align="middle" /></p>

  > [!tip]
  > 由于离散傅里叶变换具有「共轭对称性」，上面的输出结果其实是被重复了`3`次。具体结果只需看「左上角矩形」就行，其余的都是重复。
  > <p style="text-align:center;"><img src="../../image/computerVision/dftCoordination.jpg" width="25%" align="middle" /></p>

- **频谱图中心化**

  ```python
  # 频谱中心化
  shiftA = np.fft.fftshift(A)
  ```
  **作用**：挪动四个范围的频谱，让低频区域在图像中心，方便「滤波」操作。
  <p style="text-align:center;"><img src="../../image/computerVision/dftShift.jpg" width="50%" align="middle" /></p>


## 傅里叶滤波

- **思路：** 
  1. 对图像灰度进行傅里叶变换，得到频域结果
  2. 将要删除的频率所对应的傅里叶变换结果全部置为 $0 + i0$
  3. 对修改后的傅里叶变换结果进行傅里叶反变换

- **低通滤波：** 将低频部分的结果全置为零

  <p style="text-align:center;"><img src="../../image/computerVision/lowPass.jpg" width="50%" align="middle" /></p>

  <details>
  <summary><span class="details-title">Python代码</span></summary>
  <div class="details-content"> 

    ```python
      # %% 低通滤波
      import cv2
      import numpy as np

      # 图片读取
      img = cv2.imread('./cat.jpeg')
      yuv =cv2.cvtColor(img,cv2.COLOR_BGR2YUV)

      # 将灰度值转浮点类型，傅里叶变换并中心化
      yfloat = np.float32(yuv[:,:,0])
      dft = cv2.dft(yfloat,flags=cv2.DFT_COMPLEX_OUTPUT)
      dftShift = np.fft.fftshift(dft)

      # 找到低频起始，中心化后频谱的中心位置
      centerRow = int(dftShift.shape[0] / 2)
      centerCol = int(dftShift.shape[1] / 2)

      # NOTE - 高频处置为零，低频保留，然后清除对应频率幅值
      mask = np.zeros(dftShift.shape,dtype=np.uint8)
      mask[centerRow-50:centerRow+50,centerCol-50:centerCol+50,:] = 1
      dftShift = dftShift * mask

      # 反去中心。反傅里叶
      dft = np.fft.ifftshift(dftShift)
      idft = cv2.idft(dft)

      # NOTE - 傅里叶变换结果仍然是一个复数，还要转为实数，
      # 并且还要将浮点型映射为为（0 ~ 255）之间的 uint8 类型
      iyDft = cv2.magnitude(idft[:,:,0],idft[:,:,1])
      iy = np.uint8(iyDft/iyDft.max() * 255)

      # 还原图片,还原颜色通道
      yuv[:,:,0] = iy
      imgRes = cv2.cvtColor(yuv,cv2.COLOR_YUV2BGR)

      cv2.imshow('low pass',np.hstack((img,imgRes)))
      cv2.waitKey(0)
      cv2.destroyAllWindows()
    ``` 

  </div>
  </details>

- **高通滤波：** 将高频部分的结果全置为零

  <p style="text-align:center;"><img src="../../image/computerVision/highPass.jpg" width="50%" align="middle" /></p>

  <details>
  <summary><span class="details-title">Python代码</span></summary>
  <div class="details-content"> 

  ```python
    # %% 高通滤波
    import cv2
    import numpy as np

    # 图片读取
    img = cv2.imread('./cat.jpeg')
    yuv =cv2.cvtColor(img,cv2.COLOR_BGR2YUV)

    # 将灰度值转浮点类型，傅里叶变换并中心化
    yfloat = np.float32(yuv[:,:,0])
    dft = cv2.dft(yfloat,flags=cv2.DFT_COMPLEX_OUTPUT)
    dftShift = np.fft.fftshift(dft)

    # 找到低频起始，中心化后频谱的中心位置
    centerRow = int(dftShift.shape[0] / 2)
    centerCol = int(dftShift.shape[1] / 2)

    # NOTE - 低频处置为零，高频保留，然后清除对应频率幅值
    mask = np.ones(dftShift.shape,dtype=np.uint8)
    mask[centerRow-50:centerRow+50,centerCol-50:centerCol+50,:] = 0
    dftShift = dftShift * mask

    # 反去中心。反傅里叶
    dft = np.fft.ifftshift(dftShift)
    idft = cv2.idft(dft)

    # NOTE - 傅里叶变换结果仍然是一个复数，还要转为实数，并且还要将浮点型映射为为（0 ~ 255）之间的 uint8 类型
    iyDft = cv2.magnitude(idft[:,:,0],idft[:,:,1])
    iy = np.uint8(iyDft/iyDft.max() * 255)

    # 还原图片,还原颜色通道
    yuv[:,:,0] = iy
    imgRes = cv2.cvtColor(yuv,cv2.COLOR_YUV2BGR)

    cv2.imshow('low pass',np.hstack((img,imgRes)))
    cv2.waitKey(0)
    cv2.destroyAllWindows()
  ``` 
  
  </div>
  </details>

> [!tip]
> - 高通滤波：增强边缘
> - 低通滤波：模糊图片