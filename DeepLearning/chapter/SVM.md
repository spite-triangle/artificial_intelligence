# 支持向量机

# 1. 概念介绍

- **支持向量机**（support vector machines, SVM）：一种二分类模型，它的基本模型是定义在特征空间上的间隔最大的线性「分类器」，间隔最大使它有别于感知机；SVM还包括「核」技巧，这使它成为实质上的非线性分类器。

- **分类基本思想：** 根据高中数学知识，可以知道在二维平面上画一直线 $f(x,y)=0$ ，就能将平面划分为两个部分：直线上方 $f(x,y) > 0$ ；直线下方 $f(x,y) < 0$。**根据这个特性就能通过一条直线将平面的内的数据点 $(x,y)$ 划分为两个区域，即实现了分类。** 
    <p style="text-align:center;"><img src="./artificial_intelligence/image/neuralNetwork/svm_idea.jpg" width="50%" align="middle" /></p>


- **支持向量机思想：** SVM 的目标是对未知的输入数据也实现分类，这就使得「超平面」的鲁棒性要好。因此，SVM 在「决策超平面」的基础上，引入了「正超平面」与「负超平面」，正负超平面与决策超平面之间就形成了两个缓冲区域。决策超平面两侧的缓冲区域越宽，就说明数据的差异越大，那么利用决策超平面对数据进行分类时，准确度也越高。

    <p style="text-align:center;"><img src="./artificial_intelligence/image/neuralNetwork/svm_plane.png" width="50%" align="middle" /></p>

- **超平面：** 二维数据通过直线能分为两部分；三维数据能通过平面划分为两部分。对数据进行划分的界限就称之为「超平面」。
    $$
    W^TX + b = 0, \quad W = [w_1,\dotsm,w_n]^T,\quad X = [x_1,\dotsm,x_n]^T
    $$

  - **正/负超平面**：将决策超平面在垂直方向上，向上或向下移动距离 $c$。
    $$
    \begin{aligned}
        W^TX + b = + c \\
        W^TX + b = - c
    \end{aligned}
    $$

    由于 $c$ 是上下移动距离，$c=0$ 就没啥卵用了，因此得

    $$
    \begin{aligned}
    \frac{W^TX + b}{c} = +1 \\
    \frac{W^TX + b}{c} = -1 \\
    \end{aligned}
    $$

    系数除以系数，还是系数，因此可以将 $c$ 与其他系数合并

    $$
    \begin{aligned}
    W^TX + b = + 1 \\
    W^TX + b = - 1 
    \end{aligned}
    $$

> [!tip] 
> 现在关于 SVM 的模型已经知道了，实现 SVM 就需要通过样本数据求解出系数 $W,b$。**然而，超平面的取法可以有多种情况，我们的求解目的肯定是想找到最优的，因此，求解 $W,b$ 肯定是一个寻优问题。**

# 2. 硬间隔模型

## 2.1 模型介绍

- **思想：** 为了更好的区分两边的数据，就需要决策超平面与正负超平面之间的距离（间隔）最大。**最好的情况就是，正负超平面就在两边样本值的边界上，决策超平面就在正负超平面的中间，这样就能实现间隔 $d$ 最大。** 
    <p style="text-align:center;"><img src="./artificial_intelligence/image/neuralNetwork/SVM_hardMargin.png" width="50%" align="middle" /></p>
- **支持向量：** 处于正/负超平面上的样本点。

- **优化目标：** 找出 $W,b$ 使得间隔距离 $d$ 最大。**从图中也可以看出，距离 $d$ 取决于样本中的「支持向量」**

## 2.2 目标函数

**寻优目标是使得 $d$ 最大化，因此首先得求解出 $d$。**

<p style="text-align:center;"><img src="./artificial_intelligence/image/neuralNetwork/SVM_d.png" width="50%" align="middle" /></p>

假设在正/负超平面上取得两个支持向量分别为 $X_m$ 、 $X_n$，则正负超平面满足

$$
\begin{aligned}
    W^T X_m + b = - 1 \\
    W^T X_n + b = + 1 \\
\end{aligned}
$$

两式子相减

$$
W^T (X_n - X_m) = 2
$$

根据向量乘法有

$$
W^T (X_n - X_m) = ||W|| \ ||X_n - X_m|| \cos \theta
$$

从图上可知

$$
d = ||X_n - X_m|| \cos \theta
$$

最后得到

$$
d = \frac{2}{||W||}
$$

要使得 $d$ 最大，也就是使得 $||W||$ 最小，优化目标函数就可以定义为

$$
\rm{min}: \ f(W,b) = \frac{||W||^2}{2}
$$

目标函数的最优解，就是我们需要的 SVM 模型系数 $W,b$


## 2.3 约束条件

**确保所找到的超平面的有效性，就需要将两类数据分别限制在正负超平面的两边。**

<p style="text-align:center;"><img src="./artificial_intelligence/image/neuralNetwork/SVM_constraint.png" width="50%" align="middle" /></p>

对两边的样本建立标签值
- $y = -1$ 时，为负超平面一方的样本，即满足 $W^T X + b \le - 1$
- $y = +1$ 时，为正超平面一方的样本，即满足 $W^T X + b \ge + 1$

整合上面两种情况，最后约束方程就为

$$
y(W^T X + b) \ge 1
$$

## 2.4 优化模型
> <a href="https://www.bilibili.com/video/BV1HP4y1Y79e" class="jump_link"> 拉格朗日乘数法大礼包 </a>
> - 拉格朗日乘数：`00:00`
> - 对偶函数、对偶问题：`13:26`
> - 凸集、凸函数、仿射集：`19:00`
> - 弱对偶、强对偶：`25:55`
> - slater条件、KKT条件：`36:42`

### 2.4.1 模型转化

**问题模型**：

$$ 
 \begin{aligned}
    \rm{min}:& \ f(W,b) = \frac{||W||^2}{2} \\
    st:& \ g_i(W,b) = 1 - y_i(W^T X_i + b) \le 0
\end{aligned}  
$$

其中变量为 $W,b$

**从问题模型，可以看出这是一个「凸优化问题」**。利用「拉个朗日乘数法」对该问题进行求解，将上式约束问题改写为拉格朗日形式

$$
 L(W,b,\lambda) = f(W,b) + \sum_i \lambda_i g_i(W,b)
$$

其中

$$
\left \{ 
    \begin{aligned}
        \lambda_i = 0 \quad g_i(W,b) < 0 \\
        \lambda_i > 0 \quad g_i(W,b) = 0
    \end{aligned}
\right .
$$

上述拉格朗日乘数的「对偶函数」

$$
h(\lambda) = \min_{W,b} L(W,b,\lambda) \quad \lambda_i \ge 0
$$

进一步 **「对偶问题」** 就为

$$
\begin{aligned}
\max:& \ h(\lambda) \\
st:& \ \lambda_i \ge 0
\end{aligned}
$$

又由于原问题是「凸优化问题」，同时原问题的约束是仿射线性约束，进一步满足「slater条件」，**因此，$h(\lambda)$ 与 $f(W,b)$ 是「强对偶」**

$$
h(\lambda^*) = f(W^*,b^*)
$$

<span style="color:red;font-weight:bold"> 通过强对偶关系，就实现了原优化问题到对偶问题的转换。 </span>

### 2.4.2 对偶问题

求解对偶函数

$$
h(\lambda) = \min_{W,b} L(W,b,\lambda) \quad \lambda_i \ge 0
$$

求解偏导

$$
\begin{aligned}
    \frac{d L}{d W } &= W - \sum_i \lambda_i y_i X_i \\
    \frac{d L}{d b}  &= -\sum_i \lambda_i y_i \\
\end{aligned}
$$

使得 $L(W,b,\lambda)$ 最小的 $W，b$ 满足

$$
\begin{aligned}
    W - \sum_i \lambda_i y_i X_i &= 0 \\
    \sum_i \lambda_i y_i &= 0 \\
\end{aligned}
$$

将上述式子回代得

$$
\begin{aligned}
    L(W,b,\lambda) &= \frac{||\sum_i \lambda_i y_i X_i||^2}{2} + \sum_i \lambda_i [ 1 - y_i(\sum_j \lambda_j y_j X_j)^T X_i - y_i b ] \\
    &= \frac{1}{2} \sum_i \sum_j \lambda_i \lambda_j y_i y_j X_i^T X_j + \sum_i \lambda_i - \sum_i \sum_j \lambda_i \lambda_j y_i y_j X_i^T X_j - b \sum_i \lambda_i y_i  \\
    &= \sum_i \lambda_i - \frac{1}{2} \sum_i \sum_j \lambda_i \lambda_j y_i y_j X_i^T X_j 
\end{aligned}
$$

对偶问题就为

$$
\begin{aligned}
\max: \ & h(\lambda) = \sum_i \lambda_i - \frac{1}{2} \sum_i \sum_j \lambda_i \lambda_j y_i y_j X_i^T X_j  \\
st:& \ \lambda_i \ge 0 \\
& \sum_i \lambda_i y_i = 0
\end{aligned}
$$

其中

$$
\left \{ 
    \begin{aligned}
        \lambda_i = 0 \quad g_i(W,b) < 0 \\
        \lambda_i > 0 \quad g_i(W,b) = 0
    \end{aligned}
\right .
$$

**求解「对偶问题」，就只用考虑「支持向量」。**

## 2.5 模型求解

利用「 <a href="https://www.bilibili.com/video/BV1DA4y1S7XD" class="jump_link"> 序列最小优化算法 </a> 」求解「对偶问题」得到最优解 $\lambda^*$，然后回代求得

$$
W^* = \sum_i \lambda_i^* y_i X_i 
$$

再利用 $W^*$ 与 「支持向量」反解 $b^*$

$$
g_i (W^*,b^*) = 0
$$

# 3. 核技巧

## 3.1 概念

<p style="text-align:center;"><img src="./artificial_intelligence/image/neuralNetwork/SVM_nolineCategory.png" width="25%" align="middle" /></p>

<p style="text-align:center;"><img src="./artificial_intelligence/image/neuralNetwork/SVM_kernel.png" width="50%" align="middle" /></p>

- **思想：** 在二维平面，非线性的分类问题无法再通过二维的超平面进行分割，**但是，将二维数据升维变为三维数据，这样就能通过三维空间中的超平面，实现对数据的分类。**

- **维度变换：** 对原来的样本数据 $X_i$ 通过维度变换等到高纬度的数据 $T(X_i)$ 。其中变换函数为 $T()$ 。

    变换后的对偶问题

    $$
    \begin{aligned}
    \max: \ & h(\lambda) = \sum_i \lambda_i - \frac{1}{2} \sum_i \sum_j \lambda_i \lambda_j y_i y_j T(X_i)^T T(X_j)  \\
    st:& \ \lambda_i \ge 0 \\
    & \sum_i \lambda_i y_i = 0
    \end{aligned}
    $$

- **核函数：** 上述变换函数 $T(X_i)^T T(X_j)$ 的计算，首先要分别计算 $T(X_i)、 T(X_j)$，然后再计算  $T(X_i)^T T(X_j)$ 得到结果。**这一套流程下来即浪费空间，又会产生大量计算，那为何不使用一个函数直接得到上面的结果**

    $$
    K(X_i,X_j) = T(X_i)^T T(X_j)
    $$

    **其中 $K()$ 就被称之为「核函数」。** 引入核函数后，对偶问题又变为了

    $$
    \begin{aligned}
    \max: \ & h(\lambda) = \sum_i \lambda_i - \frac{1}{2} \sum_i \sum_j \lambda_i \lambda_j y_i y_j K(X_i, X_j)  \\
    st:& \ \lambda_i \ge 0 \\
    & \sum_i \lambda_i y_i = 0
    \end{aligned}
    $$

## 3.2 多项式核函数

$$
K(X_i, X_j) = (c+X_i^T X_j)^d
$$

对于 $c,d$ 系数选取不同的值，可以产生不同的维度结果。

当 $c=1,d=2$ 时，就实现了将一个二维数据转换成了一个六维的数据

$$
\begin{aligned}
   & X = [x_1,x_2]^T \\
   & Y = [1,\sqrt{2}x_1,\sqrt{2}x_2,x_1^2,x_2^2,\sqrt{2}x_1 x_2]^T \\
    满足: &\\  
   & (1 + X_i^T X_j)^2 = Y_i^T Y_j
\end{aligned}
$$

**其中，$c$ 值不要选择 $0$ 。**

<p style="text-align:center;"><img src="./artificial_intelligence/image/neuralNetwork/SVM_polynomialKernel.png" width="75%" align="middle" /></p>

## 3.3 高斯核函数

> [!warning]
> 高斯。。。又来了 (⊙﹏⊙)

$$
K(X_i, X_j) = e^{-\gamma ||X_i - X_j||^2}
$$



- **含义：** 高斯核函数描述了样本 $X_i、X_j$ 的相似程度，样本越相似 $K(X_i, X_j)$ 的值越大。
  - $\gamma$：控制样本值 $X_i、X_j$ 要多靠近，才能产生较高的相似度。$\gamma$ 越大，$X_i、X_j$ 要足够靠近，才会评判两个样本有较高相似度。

<p style="text-align:center;"><img src="./artificial_intelligence/image/neuralNetwork/SVM_GaussKernelGamma.png" width="50%" align="middle" /></p>


- **特点：** <span style="color:red;font-weight:bold"> 高斯核函数可以将，转换后的维度扩展到无限。 </span> 

<p style="text-align:center;"><img src="./artificial_intelligence/image/neuralNetwork/SVM_GaussInfinite.png" width="75%" align="middle" /></p>

# 4. 软间隔模型

## 4.1 模型介绍

<p style="text-align:center;"><img src="./artificial_intelligence/image/neuralNetwork/SVM_softMargin.png" width="50%" align="middle" /></p>

- **硬间隔的缺陷：** 在硬间隔中，我们认为所有的样本点都没有误差，但是，**当样本点中存在误差时（如图所示产生损失的黄点），就会直接导致正/负超平面间隔的减少（正/负超平面的位置取决于「支持向量」，即边界样本点）。**

- **软间隔：** 进行 SVM 建模时，需要过滤掉误差样本点的影响，然后再进行超平面寻优，这样得到的正负超平面之间的间隔，称之为软间隔。

## 4.2 铰链损失函数

<p style="text-align:center;"><img src="./artificial_intelligence/image/neuralNetwork/SVM_error.png" width="50%" align="middle" /></p>

- 有误差点的损失程度：根据约束条件可知，当样本点不在自己的分类区域时 
    $$
    y_i (W^TX_i + b) - 1 < 0
    $$
    
    因此就利于约束条件的偏离程度来定义损失值

    $$
    \varepsilon = 1 - y_i (W^T X_i + b)
    $$

- 无误差点的损失程度：

    $$
    \varepsilon = 0
    $$

综上，得到最终的样本点「铰链损失函数」

$$
\varepsilon = \max (0,1 - y_i (W^T X_i + b))
$$

## 4.3 模型求解

在硬间隔模型中，引入损失函数

$$ 
 \begin{aligned}
    \rm{min}:& \ f(W,b) = \frac{||W||^2}{2} + c \sum_i \varepsilon_i \\
    st:& \ g_i(W,b) = 1 - y_i(W^T X_i + b) - \varepsilon_i \le 0 \\
    & \varepsilon_i \ge 0
\end{aligned}  
$$

其中 $c$ 控制着对误差样本点容忍程度，$c$ 越大，就认为误差样本点越少。然后写出该该问题拉格朗日乘数形式

$$
L(W,b,\varepsilon,\lambda,u) = \frac{||\sum_i \lambda_i y_i X_i||^2}{2} + \sum_i \lambda_i g_i(W,b) - \sum_i u_i \varepsilon_i
$$

其中，将损失 $\varepsilon$ 当作了一个变量。获取原问题的对偶问题

$$
\begin{aligned}
\max: \ & h(\lambda) = \sum_i \lambda_i - \frac{1}{2} \sum_i \sum_j \lambda_i \lambda_j y_i y_j K(X_i, X_j)  \\
st:& \ \lambda_i \ge 0 \\
& \sum_i \lambda_i y_i = 0 \\
& c = \lambda_i + u_i
\end{aligned}
$$

求解方法就和硬间隔问题一样了。

> [!tip]
> 由于在约束 $g(W,b)$ 中引入了 $\varepsilon$ ，这就导致 $(W,b)$ 的系数值取决于「支持向量」与「误差样本」。

# 5. 多分类问题

-  **成对分类**：两两之间建立一个 SVM。例如，区分A、B、C数据，则 A 与 B 建立一个分类模型；B 与 C 建立一个分类模型；C 与 A 建立一个分类模型。
-  **一类对余类**：一个类型建立一个 SVM。例如，区分A、B、C数据，则 A 与 「其他」 建立一个分类模型；B 与 「其他」 建立一个分类模型；C 与 「其他」 建立一个分类模型。



