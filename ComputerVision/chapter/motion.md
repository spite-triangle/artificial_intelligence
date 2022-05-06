# 运动检测
<a href="https://github.com/spite-triangle/artificial_intelligence/tree/master/example/computerVision/motion" class="jump_link"> 本章测试程序 </a>
# 背景建模

## 帧差法

- **思想：** 由于场景中的目标在运动，目标的影像在不同图像帧中的位置不同。该类算法对时间上连续的两帧图像进行差分运算，不同帧对应的像素点相减，判断灰度差的绝对值，当绝对值超过一定阈值时，即可判断为运动目标，从而实现目标的检测功能。

- **实现：** 
    1. 相邻两张图片做差值
        $$
        D_{n}(x, y)=\left|I_{n}(x, y)-I_{n-1}(x, y)\right| \\
        $$
    2. 标记出灰度变化大于阈值的部分
        $$
        \begin{array}{l}
        R_{n}(x, y)=\left\{\begin{array}{l}
        255, \quad D_{n}(x, y)>Threshold \\
        0, \quad \text { other }
        \end{array}\right.
        \end{array}
        $$
    3. $R_{n}$ 标记出来的部分就认为就是非背景部分

- **缺陷：** 
  - **容易引入噪点**：<span style="color:red;font-weight:bold"> 摄像机稍微动一下，前后背景的像素位置就发生变化 </span> 
  - 输出结果具有空洞

```python
bgimg = cv2.imread('./asset/diffFrame_background.jpg')
fgimg = cv2.imread('./asset/diffFrame_people.jpg')
bgimgGray = cv2.cvtColor(bgimg,cv2.COLOR_BGR2GRAY)
fgimgGray = cv2.cvtColor(fgimg,cv2.COLOR_BGR2GRAY)

# 灰度差值运算
diff = np.abs(cv2.subtract(bgimgGray,fgimgGray))

# 标记灰度变化较大的部分
mask = np.zeros_like(diff)
mask[diff > 12] = 255
```

<p style="text-align:center;"><img src="../../image/computerVision/diffFrame_example.jpg" width="75%" align="middle" /></p>

## 混合高斯模型

### 理论

- **高斯混合模型：** 是由 `K` 个单高斯模型组合而成的模型，这 `K` 个子模型是混合模型的隐变量（Hidden variable）。
    - **高斯模型：** 一组数据的分布遵循高斯分布

- **背景建模思路：** <span style="color:blue;font-weight:bold">一定时间范围内，位置固定的摄像机视角下， </span> 背景可以视为不变，其灰度值就可以建立一个高斯混合模型，对于图片中，不属于背景高斯混合模型的灰度，就视为非背景部分。

- **模型建立步骤：** 
    1. 通过第一个灰度值 $I_0$，初始化第一个高斯分布 $N \sim (\mu_0 =I_0,\sigma_0)$，其中 $\sigma_0$ 初始值自己设定
    2. 获取第二个灰度值 $I_1$
        - $|I_1 - I_0| \le 3 \sigma_0$：该灰度值属于高斯模型 $N \sim (\mu_0,\sigma_0)$，并对高斯模型进行更新
        - $|I_1 - I_0| > 3 \sigma_0$：该灰度值不属于 $N \sim (\mu_0,\sigma_0)$，则需要新建一个高斯模型 $N \sim (\mu_1=I_1,\sigma_1)$，其中 $\sigma_1$ 初始值自己设定
    3. 重复上述步骤，将所有的高斯模型都建立出来。

- **模型背景检测：** 输入的灰度值与混合高斯模型中的每一个均值做差，如果其差值在`2`倍的方差之间的话，则认为是背景，否则认为是前景。将前景赋值为`255`,背景赋值为`0`。这样就形成了一副前景二值图

### 案例

```python
# 高斯混合模型
GaussModel = cv2.createBackgroundSubtractorMOG2()
# 高斯混合
mask = GaussModel.apply(img)
```

<p style="text-align:center;"><img src="../../image/computerVision/GaussModel.gif" width="50%" align="middle" /></p>
