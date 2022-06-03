# 策略学习


# 算法思路

- **目标**：控制动作执行的概率，使得最终的「状态激活函数」的结果最大，即动作执行后，使得平均局势最优（ $E_S[V_{\pi}(s)]$ 值最大 ）。
- **策略**：得到「策略函数 $\pi(a|s)$」，这样就能通过带入状态 $s$，**根据动作的发生的概率随机选择动作**。
- **问题**：要使得局势最优，就要知道每种状态下最优的 $\pi(a|s)$ ，但是角色无法认知。
- **解决**：利用一个神经网络 $\pi(a|s;\theta)$ 来近似拟合出 每种状态下应当采取的 $\pi(a|s)$
  - `s`：神经网络的输入
  - `a`：神经网络的输出
  - $\theta$：神经网络的参数 

# 神经网络的构造

<p style="text-align:center;"><img src="../../image/reinforceLearning/example_policy_idea.jpg" width="75%" align="middle" /></p>

1. 一张游戏画面当作一个「状态」，状态通过「卷积层」实现特征提取
2. 特征输入给「权连接层」进行 $\pi(a|s)$ 拟合
3. 将「权连接层」的输出通过`softmax`层转化为概率
4. 输出每个动作在当前状态下产生好结果的概率

# 策略梯度训练

1. **目标函数**：
    $$
    \begin{aligned}
    V(s_t;\theta) &= E_A [ Q_\pi (s_t,A) ] = \sum\limits_a \pi(a|s_t;\theta) Q_\pi (s_t,a) \\
    J(\theta) &= E_S[V(S;\theta)]
    \end{aligned}
    $$

2. **参数更新**：<span style="color:red;font-weight:bold"> 由于 $E_S[V(S;\theta)]$ 不好计算，所以使用 $V(S;\theta)$ 来近似；由于求解的是 $J(\theta)$ 最大，所以使用的是`+`</span>
    $$
    \theta = \theta + \beta \frac{\partial V(s;\theta)}{\partial \theta}
    $$

# 策略梯度算法

**定义：** $\frac{\partial V(s;\theta)}{\partial \theta}$ 就是策略梯度。

**计算：** 将策函数关于$\theta$求导展开

$$
\begin{aligned}
\frac{\partial V(s;\theta)}{\partial \theta} &= \frac{\partial \sum\limits_a \pi(a|s;\theta) Q_\pi (s,a)}{\partial \theta} \\
&= \sum\limits_a (\frac{\partial \pi(a|s;\theta)}{\partial \theta} Q_\pi (s,a) + \frac{\partial Q_\pi (s,a)}{\partial \theta} \pi(a|s;\theta))
\end{aligned}
$$

由于 $Q_\pi (s,a)$ 求解关于 $\theta$ 的导数也困难，所以忽略掉

$$
\frac{\partial V(s;\theta)}{\partial \theta} = \sum\limits_a \frac{\partial \pi(a|s;\theta)}{\partial \theta} Q_\pi (s,a)
$$

对于上述公式只能求解离散形式，连续形式可以变形为：

$$
\begin{aligned}
\frac{\partial V(s;\theta)}{\partial \theta} &= \int_a \pi(a|s;\theta) \frac{\partial \ln \pi(a|s;\theta)}{\partial \theta} Q_\pi (s,a) \ d a\\
&= E_A [\frac{\partial \ln \pi(a|s;\theta)}{\partial \theta} Q_\pi (s,a)]
\end{aligned}
$$

可以利用 [蒙特卡洛积分](./ReinforcementLearning/chapter/monte_carlo_withNum.md) 对上述期望进行近似计算。

1. 根据策略函数 $\pi (a|s;\theta)$ 随机获取动作 $\hat a$
2. 带入 $\hat a$ 计算：$g(\hat a,\theta) = \frac{\partial \ln \pi(\hat a|s;\theta)}{\partial \theta} Q_\pi (s,\hat a)$
3. 然后通过 $g(\hat a,\theta)$ 来近似计算：$\frac{\partial V(s;\theta)}{\partial \theta} =E_A[g(A,\theta)] $

# 价值函数计算

- **方法一：** 角色游玩一局游戏后，再计算一次 $Q_\pi (s_t,a_t)$ ，然后用该值来近似代替，这样就导致一局游戏才能更新一次$\theta$系数。
- **方法二：** 对 $Q_\pi (s_t,a_t)$ 再建立一个神经网络
