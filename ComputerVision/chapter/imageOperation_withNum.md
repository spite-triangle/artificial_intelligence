# 图像进阶操作

<a href="https://github.com/spite-triangle/artificial_intelligence/tree/master/example/computerVision/imageOperation" class="jump_link"> 本章测试程序 </a>

# 1. 图像运算

## 1.1. 加减乘除

> [!tip]
> 由之前的章节，已经明确图片的本质就是「矩阵（三维数组）」，所以可以对图片进行数学运算。

- **图像相加：**

    ```python
    # 图像直接相加
    imgA + imgB
    cv2.add(imgA,imgB)
    cv2.addWeighted(imgA, alpha, imgB, beta, gamma)
    ```

  - `imgA + imgB`：**当数值大于一个字节时，大于一个字节的位数都被丢失了**。
    $$
    (A + B) \ \% \ 256
    $$

  - `cv2.add(imgA,imgB)`：**当数值超过`255`时，取值为`255`**
    $$
    \min(A+B,255)
    $$
   
  - `cv2.addWeighted(imgA, alpha, imgB, beta, gamma)`：
    $$
    \rm min(round(A*\alpha + B *\beta + \gamma),255) 
    $$

- **图像相乘：** 规则与相加类似


## 1.2. 位运算

```python
# 与运算
bitwise_and(src1:image, src2:image[, dst[, mask]]) -> dst
# 或运算
bitwise_or(src1:image, src2:image[, dst[, mask]]) -> dst
# 异或运算
bitwise_xor(src1:image, src2:image[, dst[, mask]]) -> dst
# 非
bitwise_not(src1:image[, dst[, mask]]) -> dst
```

- **与、或、异或：** 实质就是两个图像数组，相同位置的数据直接进行与、或、异或运算。
  - **与：** 图片亮度会整体变暗，与运算会将值变小，不超过`255`
  - **或：** 图片亮度会整体变亮，与运算会将值变大，不超过`255`
- **非：** 与程序中按位取反不一样，OpenCV 中实现的是对颜色反转
    $$
    dst = 255 - src
    $$
    <p style="text-align:center;"><img src="/artificial_intelligence/image/computerVision/bitwise_not.jpg" width="50%" align="middle" /></p>

# 2. 翻转与旋转

## 2.1. 翻转

```python
# 翻转
# flip(src, flipCode[, dst]) -> dst
# flipCode = 0：垂直翻转
# flipCode < 0：垂直 + 水平翻转
# flipCode > 0：水平翻转
img0 = cv2.flip(img,0)
imgLow0 = cv2.flip(img,-1)
imgGreat0 = cv2.flip(img,1)
```

<p style="text-align:center;"><img src="/artificial_intelligence/image/computerVision/flip.jpg" width="75%" align="middle" /></p>

## 2.2. 旋转

**作用：** 以图片中心，对图片进行旋转，角度只能为 180，顺时针 90，逆时针90。
```python
# 旋转
# rotate(src, rotateCode[, dst]) -> dst
# roteCode：cv2.ROTATE_
imgr = cv2.rotate(img,cv2.ROTATE_180)
```

> [!tip]
> 翻转不会改变原来图片的`np.ndarray.shape`，旋转会修改。

# 3. 仿射变换

## 3.1. 介绍

仿射变换就是对图片的像素坐标进行位置变换，进而实现对图片的平移、旋转和缩放。对于图片中的像素会定义一个坐标系，该坐标系以横向像素为`x`轴，高度像素为`y`轴，坐标原点为图片左上角。
<p style="text-align:center;"><img src="/artificial_intelligence/image/computerVision/pixelCoordinates.png" width="50%" align="middle" /></p>

仿射变换中集合中的一些性质保持不变：
- **共线性**：若几个点变换前在一条线上，则仿射变换后仍然在一条线上
- **平行性**：若两条线变换前平行，则变换后仍然平行
- **共线比例不变性**：变换前一条线上的两条线段的比例在变换后比例不变

## 3.2. 变换的数学表达

### 3.2.1. 平移

将原图的像素坐标进行移动，其数学表达就为
$$
\begin{aligned}
x = x_c + t_x \\
y = y_c + t_y 
\end{aligned}
$$

$(x_c,y_c)$ 为原图像素的坐标，$(x,y)$为变换后的像素坐标，$(t_x,t_y)$移动的距离。将上面公式改写成矩阵形式

$$
\begin{bmatrix}
  x \\
  y 
\end{bmatrix} = 
\begin{bmatrix}
  1 & 0 \\
  0 & 1 
\end{bmatrix} \begin{bmatrix}
  x_c \\
  y_c 
\end{bmatrix} + \begin{bmatrix}
  t_x \\
  t_y
\end{bmatrix}
$$

### 3.2.2. 缩放

将原图的像素坐标进行缩放，其数学表达就为
$$
\begin{aligned}
x = \alpha_x x_c  \\
y = \alpha_y y_c  
\end{aligned}
$$
其矩阵形式为
$$
\begin{bmatrix}
  x \\
  y 
\end{bmatrix} = 
\begin{bmatrix}
  \alpha_x & 0 \\
  0 & \alpha_y
\end{bmatrix} \begin{bmatrix}
  x_c \\
  y_c 
\end{bmatrix}
$$

### 3.2.3. 旋转

<p style="text-align:center;"><img src="/artificial_intelligence/image/computerVision/rotation_transform.jpg" width="50%" align="middle" /></p>

像素点$P_c(x_c,y_c)$ 点逆时针旋转 $\theta$ 角，旋转到 $P(x,y)$  位置。可以当作是坐标系 $Ox_c y_c$ 逆时针旋转 $- \theta$ 角后，$P_c$ 像素点在 $Oxy$ 坐标系的位置，因此可以推导出坐标变换
$$
\begin{aligned}
  x &= x_c \cos(-\theta)  + y_c \sin(- \theta)\\
  y &= - x_c \sin(-\theta) + y_c \cos(- \theta)
\end{aligned}
$$

化简得到矩阵形式

$$
\begin{bmatrix}
  x \\
  y 
\end{bmatrix} = 
\begin{bmatrix}
  \cos(\theta) & - \sin(\theta) \\
  \sin(\theta) & \cos(\theta)
\end{bmatrix} \begin{bmatrix}
  x_c \\
  y_c 
\end{bmatrix}
$$


## 3.3. 变换矩阵

上一小节推导了平移、缩放、旋转的数学表达公式：

$$
\begin{bmatrix}
  x \\
  y 
\end{bmatrix} = 
\begin{bmatrix}
  1 & 0 \\
  0 & 1 
\end{bmatrix} \begin{bmatrix}
  x_c \\
  y_c 
\end{bmatrix} + \begin{bmatrix}
  t_x \\
  t_y
\end{bmatrix}
$$


$$
\begin{bmatrix}
  x \\
  y 
\end{bmatrix} = 
\begin{bmatrix}
  \alpha_x & 0 \\
  0 & \alpha_y
\end{bmatrix} \begin{bmatrix}
  x_c \\
  y_c 
\end{bmatrix}
$$

$$
\begin{bmatrix}
  x \\
  y 
\end{bmatrix} = 
\begin{bmatrix}
  \cos(\theta) & - \sin(\theta) \\
  \sin(\theta) & \cos(\theta)
\end{bmatrix} \begin{bmatrix}
  x_c \\
  y_c 
\end{bmatrix}
$$

从中可以看出缩放和旋转都能通过一个 2x2 的矩阵进行变换，而平移带有一个偏移项，为统一变换矩阵形式，就可以将平移改写为：

$$
\begin{bmatrix}
  x \\
  y \\
  1
\end{bmatrix} = 
\begin{bmatrix}
  1 & 0  & t_x \\
  0 & 1  & t_y \\
  0 & 0 & 1
\end{bmatrix} \begin{bmatrix}
  x_c \\
  y_c \\ 
  1
\end{bmatrix} 
$$

同样，也将旋转和缩放改写为 3x3 的形式：

$$
\begin{bmatrix}
  x \\
  y \\
  1 
\end{bmatrix} = 
\begin{bmatrix}
  \alpha_x & 0 & 0 \\
  0 & \alpha_y & 0 \\
  0 & 0 & 1
\end{bmatrix} \begin{bmatrix}
  x_c \\
  y_c \\
  1
\end{bmatrix}
$$

$$
\begin{bmatrix}
  x \\
  y \\
  1 
\end{bmatrix} = 
\begin{bmatrix}
  \cos(\theta) & - \sin(\theta) & 0\\
  \sin(\theta) & \cos(\theta) & 0 \\
  0 & 0 & 1  
\end{bmatrix} \begin{bmatrix}
  x_c \\
  y_c \\
  1
\end{bmatrix}
$$

**综上，旋转、平移、缩放都可以通过一个`3x3`的变换矩阵，实现坐标变换。该变换矩阵还能实现切变、对称翻转操作。**

<p style="text-align:center;"><img src="/artificial_intelligence/image/computerVision/transform.jpg" width="50%" align="middle" /></p>

## 3.4. 变换叠加

上文所的变换都是实现图像旋转、缩放、平移、切边的单步变换，通过变换矩阵就可以实现更便捷的多步变换。假设旋转的变换矩阵为`R`；平移的变换矩阵为`M`；缩放的变换矩阵为`S`，那么旋转、平移、缩放的组合变换就能写为：

$$
\begin{bmatrix}
  x \\
  y \\
  1 
\end{bmatrix} = M_k R_j S_i \dotsm R_1 M_2 S_1 M_1 
\begin{bmatrix}
  x_c \\
  y_c \\
  1
\end{bmatrix}
$$

**组合变换的变换矩阵顺序为「从右往左」，这样就能首先计算出整体的变换矩阵 $M_k R_j S_i \dotsm R_1 M_2 S_1 M_1$ 的结果，然后才进行像素的坐标变换。**

## 3.5. 变换矩阵的逆推

> [!tip]
> 逆推两张相同图片之间的「仿射变换矩阵」，需要知道`3`组对应的像素坐标点。

<p style="text-align:center;"><img src="/artificial_intelligence/image/computerVision/transform.jpg" width="50%" align="middle" /></p>

观察图中所有变换矩阵，其形式可以写为：
$$
\begin{bmatrix}
  x \\
  y \\
  1 
\end{bmatrix} =
\begin{bmatrix}
  a_{11} & a_{12} & a_{13}\\
  a_{21} & a_{22} & a_{23}\\
  0 & 0 & 1  
\end{bmatrix}
\begin{bmatrix}
  x_c \\
  y_c \\
  1
\end{bmatrix}
$$

**也就是说从图片A变换到图片B的「仿射变换矩阵」一共有`6`个未知数，而一组$(x,y),(x_c,y_c)$ 就能构造`2`组方程，所以需要`3`组对应的像素点坐标。**


## 3.6. OpenCV 代码


```python
# 逆时针旋转转变换矩阵
# center，旋转中心
# angle，逆时针旋转角度
# scale，图片缩放值
cv2.getRotationMatrix2D: (center: tuple, angle, scale) -> dst

# M ：仿射变换矩阵
# dsize ：输出图片的大小
# flags：图片的插值算法，默认算法就不错
# borderMode：查看 图像边界扩展 小节
cv2.warpAffine(src, M, dsize[, dst[, flags[, borderMode[, borderValue]]]]) -> dst

# 仿射变换矩阵逆推
# src：3x2 的 numpy.ndarray 矩阵，数据类型为 np.float
# dst：3x2 的 numpy.ndarray 矩阵，数据类型为 np.float
cv2.getAffineTransform(src, dst) -> M
```

> [!tip]
> 仿射变换矩阵`M`为`2x3`的`numpy.ndarray`矩阵且类型为`dtype =np.float`。因为最后一行都为`[0,0,1]`，所以省略了。


<p style="text-align:center;"><img src="/artificial_intelligence/image/computerVision/affineTransformResult.jpg" width="75%" align="middle" /></p>

# 4. 透视变换

## 4.1. 齐次坐标

$$
\begin{bmatrix}
  x \\
  y \\
  1 
\end{bmatrix} =
\begin{bmatrix}
  a_{11} & a_{12} & a_{13}\\
  a_{21} & a_{22} & a_{23}\\
  0 & 0 & 1  
\end{bmatrix}
\begin{bmatrix}
  x_c \\
  y_c \\
  1
\end{bmatrix}
$$

仿射变换中，用来表示「二维像素位置」的坐标为

$$
\begin{bmatrix}
  x \\
  y \\
  1 
\end{bmatrix}
$$

从形式上来说，这就是用了「三维坐标」来表示「二维坐标」，即 **降维打击**。在将`1`进行符号化，用`w`进行代替

$$
\begin{bmatrix}
  x \\
  y \\
  w
\end{bmatrix}
$$

**这种表达 `n-1` 维坐标的 `n` 维坐标，就被称之为「齐次坐标」。**

## 4.2. 透视

透视的目的就是实现 **近大远小**，也就是需要有纵向的深度，而像素位置 $(x,y)$ 只能表示像素在平面上的位置关系，此时「齐次坐标」就能排上用场了。三维的齐次坐标虽然表示的二维的平面，但是其本质还是一个三维空间的坐标值，这样就能将图片像素由「二维空间」扩展到「三维空间」进行处理，齐次坐标的`w`分量也就有了新的含义：三维空间的深度。

<p style="text-align:center;"><img src="/artificial_intelligence/image/computerVision/perspective.jpg" width="50%" align="middle" /></p>


在「仿射变换」中，像素的齐次坐标为 $[x,y,1]^T$，可以解释为图像位于三维空间 的 $w=1$ 平面上，即 $w=1$ 平面就是我们在三维空间中的视线平面（三维空间中的所有东西都被投影到 $w=1$ 平面，然后我们才能看见）。「透视」就规定了所有物体如何投影到视线平面上，即「近大远小」。数学描述就是根据像素三维空间中的坐标点 $P(x,y,w)$ 得出像素在视线平面上的坐标 $P_e(x_e,y_e,1)$，两个关系如图所示，根据三角形相似定理就能得出：

$$
\begin{aligned}
  \frac{x}{x_e} = \frac{w}{1} \\
  \frac{y}{y_e} = \frac{w}{1}
\end{aligned}
$$

整理得：

$$
\begin{aligned}
  x_e = \frac{x}{w} \\
  y_e = \frac{y}{w} \\
  1 = \frac{w}{w}
\end{aligned}
$$

上述公式就实现了三维空间像素坐标向视线平面的透视投影。

## 4.3. 透视变换

$$
\begin{bmatrix}
  x \\
  y \\
  w 
\end{bmatrix} =
\begin{bmatrix}
  a_{11} & a_{12} & a_{13}\\
  a_{21} & a_{22} & a_{23}\\
  a_{31} & a_{32} & a_{33}\\
\end{bmatrix}
\begin{bmatrix}
  x_c \\
  y_c \\
  1
\end{bmatrix}
$$

根据「仿射变换」可知，上述矩阵就能实现图片像素坐标 $[x_e,y_e,1]^T$ 在三维空间中的旋转、缩放、切变的变换操作（没有三维空间的平移，变换矩阵差一个维度），得到像素位置变换后的三维坐标就为 $[x,y,w]^T$。再将新的像素齐次坐标进行透视处理，将坐标映射到 $w=1$ 平面， 得到的像素位置就是最终「透视变换」的结果。

<p style="text-align:center;"><img src="/artificial_intelligence/image/computerVision/perspectiveTransform.jpg" width="50%" align="middle" /></p>

因此透视变换的变换矩阵就能改写为

$$
\begin{bmatrix}
  x' \\
  y' \\
  1
\end{bmatrix} = \frac{1}{w}
\begin{bmatrix}
  a_{11} & a_{12} & a_{13}\\
  a_{21} & a_{22} & a_{23}\\
  a_{31} & a_{32} & a_{33}\\
\end{bmatrix}
\begin{bmatrix}
  x_c \\
  y_c \\
  1
\end{bmatrix}
$$

由于`w`是一个常量，也可以放入变换矩阵：
$$
\begin{bmatrix}
  x' \\
  y' \\
  1
\end{bmatrix} = 
\begin{bmatrix}
  a_{11} & a_{12} & a_{13}\\
  a_{21} & a_{22} & a_{23}\\
  a_{31} & a_{32} & a_{33}\\
\end{bmatrix}
\begin{bmatrix}
  x_c \\
  y_c \\
  1
\end{bmatrix}
$$

将矩阵拆解

$$
\begin{aligned}
x' &= a_{11} x_c + a_{12} y_c + a_{13}\\
y' &= a_{21} x_c + a_{22} y_c + a_{23} \\
1 &= a_{31} x_c + a_{32} y_c + a_{33} \\
\end{aligned}
$$

根据齐次坐标透视规则

$$
\begin{aligned}
x' &= \frac{a_{11} x_c + a_{12} y_c + a_{13}}{a_{31} x_c + a_{32} y_c + a_{33}} \\
y' &= \frac{a_{21} x_c + a_{22} y_c + a_{23}}{a_{31} x_c + a_{32} y_c + a_{33}}  
\end{aligned}
$$

可以看出，对分式上下乘以一个非零常数 $\alpha$ ，值不变
$$
\begin{aligned}
x' &= \frac{ \alpha ( a_{11} x_c + a_{12} y_c + a_{13} )}{\alpha( a_{31} x_c + a_{32} y_c + a_{33} )} \\
y' &= \frac{\alpha( a_{21} x_c + a_{22} y_c + a_{23} )}{\alpha( a_{31} x_c + a_{32} y_c + a_{33} )}  
\end{aligned}
$$

再将上面的式子写成矩阵形式

$$
\begin{bmatrix}
  x' \\
  y' \\
  1
\end{bmatrix} = \alpha
\begin{bmatrix}
  a_{11} & a_{12} & a_{13}\\
  a_{21} & a_{22} & a_{23}\\
  a_{31} & a_{32} & a_{33}\\
\end{bmatrix}
\begin{bmatrix}
  x_c \\
  y_c \\
  1
\end{bmatrix}
$$

可以看出变换矩阵乘以一个非零常数，对结果无影响，那么就直接令 $\alpha = \frac{1}{a_{33}}$

$$
\begin{bmatrix}
  x' \\
  y' \\
  1
\end{bmatrix} = \frac{1}{a_{33}}
\begin{bmatrix}
  a_{11} & a_{12} & a_{13}\\
  a_{21} & a_{22} & a_{23}\\
  a_{31} & a_{32} & a_{33}\\
\end{bmatrix}
\begin{bmatrix}
  x_c \\
  y_c \\
  1
\end{bmatrix}
$$


其结果就为

$$
\begin{bmatrix}
  x' \\
  y' \\
  1
\end{bmatrix} = 
\begin{bmatrix}
  a_{11} & a_{12} & a_{13}\\
  a_{21} & a_{22} & a_{23}\\
  a_{31} & a_{32} & 1\\
\end{bmatrix}
\begin{bmatrix}
  x_c \\
  y_c \\
  1
\end{bmatrix}
$$

**可以看出两张图片之间的透视变换，只涉及`8`个自由度。**


> [!tip|style:flat]
> 从最后的公式形式可以看出，仿射变换其实就是透视变换的一种特例，仿射变换只是 $w=1$ 的平面内进行平移、缩放、旋转等。


## 4.4. 透视变换逆推

根据下式可知，一对像素坐标点 $(x_c,y_c),(x',y')$ 只能构成`2`组方程
$$
\begin{aligned}
x' &= \frac{a_{11} x_c + a_{12} y_c + a_{13}}{a_{31} x_c + a_{32} y_c + a_{33}} \\
y' &= \frac{a_{21} x_c + a_{22} y_c + a_{23}}{a_{31} x_c + a_{32} y_c + a_{33}}  
\end{aligned}
$$
然后透视变换矩阵具有`8`个未知变量，所以**逆向求解变换矩阵需要`4`对像素坐标**。
$$
\begin{bmatrix}
  x' \\
  y' \\
  1
\end{bmatrix} = 
\begin{bmatrix}
  a_{11} & a_{12} & a_{13}\\
  a_{21} & a_{22} & a_{23}\\
  a_{31} & a_{32} & 1\\
\end{bmatrix}
\begin{bmatrix}
  x_c \\
  y_c \\
  1
\end{bmatrix}
$$

## 4.5. OpenCV 代码

```python

# 逆向计算透视变换矩阵
# srcPoints : 像素点坐标
# dstPoints : 像素点坐标
cv2.getPerspectiveTransform(srcPoints:np.ndarray, dstPoints:np.ndarray[, solveMethod]) -> retval

# 透视变换
# src ：图片
# M ：透视变换矩阵 3x3
# dsize : 要显示的图片大小
cv2.warpPerspective(src:image, M, dsize[, dst[, flags[, borderMode[, borderValue]]]]) -> dst:image
```

> [!note]
> `srcPoints,dstPoints`的 `dtype` 必须写为 `np.float32`，而非`np.float、np.float`。

# 5. 阈值控制

```python
# 阈值控制
ret,destImg = cv2.threshold(img,threshVal,maxVal,flags)
```
- **作用：** 根据设定的「阈值」将图像拆分为两个部分，例如纯黑-纯白图。

- **图像阈值类型：** 
  - `cv2.THRESH_BINARY`：
    - \> threshVal：通道值大于阈值时，取值为`maxVal`
    - < threshVal：通道值大于阈值时，取值为`0`
  - `cv2.THRESH_BINARY_INV`：计算方式与上面相反
  - `cv2.THRESH_TOZERO`：
      - \> threshVal：通道值大于阈值时，不变
      - < threshVal：通道值大于阈值时，取值为`0`
  - `cv2.THRESH_TOZERO_INV`：计算方式与上面相反
  - `cv2.THRESH_TRUNC`：
      - \> threshVal：通道值大于阈值时，取值为`maxVal`
      - < threshVal：通道值大于阈值时，不变

  <p style="text-align:center;"><img src="/artificial_intelligence/image/computerVision/threshold_categories.jpg" width="50%" align="middle" /></p>


# 6. 图像边界扩展

**作用：** 当图像需要变大，但是不想直接缩放，则可以选择不同的方法将图片外围进行扩展，使得原图变大

```python
destImage = cv2.copyMakeBorder(src: Mat, 
                              top_size, bottom_size, left_size, right_size, 
                              borderType)
```
- `top_size, bottom_size, left_size, right_size`：图片上下左右，需要填充的像素值
- `borderType`：填充方式
  - `BORDER_REPLICATE`:复制法，将最边缘像素向外复制。
  - `BORDER_REFLECT`:反射法，对感兴趣的图像中的像素在两边进行复制例如：dcba | abcd（原图） | dcba
  - `BORDER_REFLECT1O1`:反射法，也就是以最边缘像素为轴，例如 dcb | abcd（原图） | cba，**没有复制最边缘的像素**
  - `BORDER_WRAP`:外包装法，abcd | abcd | abcd ，重复原图
  - `BORDER_CONSTANT`:常量法，边界用常数值填充。111 | abcd | 111

  <p style="text-align:center;"><img src="/artificial_intelligence/image/computerVision/makeBorder_categories.jpg" width="50%" align="middle" /></p>
