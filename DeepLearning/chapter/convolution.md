
# 卷积神经网络

# 一、卷积计算

<center>

![convolution](../../image/neuralNetwork/convolution.jpg)
</center>

&emsp;&emsp; 输入图片（6x6）与过滤器（3x3）进行矩阵的卷积计算，得到输出（4x4）。

输出纬度的计算：
$$
n_o = \lfloor \frac{n_{i} + 2p - f}{s} + 1\rfloor \tag{1.1}
$$

> f : 过滤器的纬度 
> p : 输入图片填充的像素
> s : 过滤器在输入图像上移动的步长
> **当步长s不为1时，可能导致过滤器越界，所以使用floor进行向下取整。**

&emsp;&emsp; <span style="color:orange;font-weight:bold"> 在图像识别中所说的“卷积”和实际定义有一点小差别，图像识别省略了：对过滤器进行右对角线的翻转（对结果没啥影响，没必要算了）。图像识别中的卷积准确应当称之为：cross-correlation。 </span>

# 二、过滤器边缘监测
<center>

![edge detection](../../image/neuralNetwork/edgedetection.jpg)
</center>

&emsp;&emsp;定义一个竖向的过滤器，就能实现对竖向的边缘进行监测；同样定义一个横向的过滤器，就能对一个横向的边缘进行监测。对于过滤器的值，可以自定义，不用完全是-1或者1。**因此，卷积神经网络就直接将过滤器中的所有系数，都设置为w系数。**

# 三、Padding

&emsp;&emsp;进行卷积计算后，原来的图像会被缩小，为了规避这个问题，可以将原来的图像的进行边缘扩充像素。根据公式（1.1）就能将卷积后的图像尺寸保持得和输入图像一样大。

# 四、三维卷积

<center>

![rgb convolution](../../image/neuralNetwork/rgbConvolution.jpg)
</center>

&emsp;&emsp;**输入的通道数和过滤器的通道数相同，才能进行计算。**

# 五、一层网络

<center>

![single layer](../../image/neuralNetwork/convolutionSingleLayer.jpg)
</center>

&emsp;&emsp; **一张图像通过两个滤波器，计算得到两层；在将两层送入激活函数；最后将两个层重合得到输出。**

$$
    a^{[l]} = active(input * filter + bias)  \tag{5.1}
$$

# 六、池化（pooling）

<center>

![pool](../../image/neuralNetwork/pooling.jpg)
</center>

&emsp;&emsp;<span style="color:blue;font-weight:bold"> 不同于卷积，池化的作用是将原来的像素，按照块操作，进行压缩处理。</span> **根据压缩数据方式的不同分为：1）max pooling，区域内的最大值；2）average pooling，区域求平均值。**

$$
n_o = \lfloor \frac{n_{i} - f}{s} + 1\rfloor \tag{6.1}
$$

<center>

![pooling cal](../../image/neuralNetwork/poolingcal.jpg)
</center>

> * **对于pooling的计算，是每一个通道进行一次计算，输出结果通道数不改变。**
> * **pooling涉及的超参数，训练集训练时就是为常量。**

# 七、卷积神经网络模型

<center>

![convolution model](../../image/neuralNetwork/convolutionModel.jpg)
</center>

> * **图像刚输入的时候，靠卷积和池化降低参数数量。**
> * **输出就是靠全连接网络（BP网络）和分类器获取估计结果。**

# 八、卷积的作用

<center>

![parameters](../../image/neuralNetwork/parameters.jpg)
</center>

&emsp;&emsp;<span style="color:blue;font-weight:bold"> 与普通神经网络相比，卷积网络大大降低了超参数的量，实现了对图像像素的压缩和特征提取。</span>

> * **parameter sharing**：用于一次过滤的过滤器都是一样的，数据处理过程统一，也降低了超参数的量。
> * **sparsity of connection**：每个像素只能影响局部的结果，符合常规认知。
