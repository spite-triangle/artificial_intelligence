# 蒙特卡洛

# 计算$\pi$

## 采样法

<p style="text-align:center;"><img src="../../image/reinforceLearning/calculating_pi_1.jpg" width="50%" align="middle" /></p>

从图中可知：矩形的面积为 $A_r = 4$，圆的面积为 $A_c = \pi$。现在在矩形框内随机的抽取采样点（**均匀抽样**），那么采样点位于圆内的概率就为 $p = \frac{A_c}{A_r} =\frac{\pi}{4}$。

**步骤：**

- 采样`m`个点，有`n`个点在圆内
- $\pi$值就是：$\pi \approx \frac{4 n}{m}$ 
- 误差精度为：$|\frac{4 n}{m} - \pi| = O(\frac{1}{\sqrt{m}})$

## Buffon针问题

<p style="text-align:center;"><img src="../../image/reinforceLearning/calculating_pi_buffon.jpg" width="50%" align="middle" /></p>

所有的针都一摸一样，针与纸相交的概率就为 $P = \frac{2l}{\pi d}$。这样根据`monte carlo`就计算 $\pi$。

# 计算定积分

**问题：** 计算下列定积分的值

$$
I = \int_a^b f(x) \ dx
$$


**定积分的定义**：

$$
I =\lim_{n \to \infin} \sum_{i=0}^n f(a + i * \frac{b - a}{n}) \frac{b-a}{n}
$$

**近似计算：**
1. 在 $[a,b]$ 内随机采样 $x_1,x_2,\dotsm,x_n$
2. 根据定积分定义进行近似计算：$I \approx (b - a) \frac{1}{n} \sum\limits_{i=1}^n f(x_i) $

# 计算期望

**期望定义：** 试验中每次可能结果的概率乘以其结果的总和，反映随机变量平均取值的大小。$p(x)$代表概率密度函数。

$$
E(f(X)) = \int f(x) p(x) \ dx
$$


**近似计算：**

1. 根据概率密度函数 $p(x)$ 随机采样 $x_1,x_2,\dotsm,x_n$
2. 根据定积分定义进行近似计算：$E(f(X)) \approx \frac{1}{n} \sum\limits_{i=1}^n f(x_i) $