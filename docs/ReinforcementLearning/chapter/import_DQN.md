# DQN优化

# 经验回放

## 概念

**传统DQN的缺陷：**
1. 传统的DQN，一次 $(s_t,a_t,r_t,s_{t+1})$ 只用于一次计算，十分浪费
2. 用于训练的 $(s_t,a_t,r_t,s_{t+1})$ 具有较强的相关性（$s_t$训练完，就使用 $s_{t+1}$），这种相关性是有害的

**经验回放（experience replay）：** 
<p style="text-align:center;"><img src="../../image/reinforceLearning/replayBuffer.jpg" width="75%" align="middle" /></p>

1. 将每一次 $(s_t,a_t,r_t,s_{t+1})$ 都放入一个「回放缓冲区`replay buffer`」中。缓冲区大小一定，存满后，就进行入队和出队
2. 利用「随机梯度`SGD`」进行模型的更新：<span style="color:red;font-weight:bold"> 从回放缓冲区中，随机抽取一项进行模型更新计算 </span>

## 优先经验回放

**原因：** 角色对于常规的游戏环境会在「回放缓冲区」中存放大量的经验，而对于BOSS关卡的经验却很少。对于经验回放，若采用均匀抽样，就会导致对BOSS关卡的学习机会变少。
<p style="text-align:center;"><img src="../../image/reinforceLearning/replayProblem.jpg" width="75%" align="middle" /></p>

**优先经验回放：** 对于特殊状态（例如BOSS关卡），增加其被抽中学习的概率。
1. **实现方法：** $|\delta_t|$ 越大，被抽中的概率越大
   - **方法一：** $p_t \propto |\delta_t| + \epsilon$，即被抽样的概率 $p_t$ 正比于TD error绝对值 $|\delta_t|$
   - **方法二：** $p_t \propto \frac{1}{i}$，其中 `i` 是状态根据 $|\delta_t|$ 降序排列的索引。

2. **学习率：** $(n p_t)^{-\beta} \alpha, \beta \in (0,1)$，即被抽中的概率越大，学习率应当越小。<span style="color:blue;font-weight:bold"> 用于消除大概率抽样，造成的预测偏差 </span>

3. **更新 $|\delta_t|$：**
   - 初始值设置为最大值。因为状态只有进行一次学习计算后才能知道结果
   - 每一次学习后，更新对于状态的 $|\delta_t|$

<p style="text-align:center;"><img src="../../image/reinforceLearning/import_replay.jpg" width="75%" align="middle" /></p>  

# 高估问题

## 高估的原因

<span style="font-size:24px;font-weight:bold" class="section2">1. 最大化</span>


**定理：** $x_1,x_2,\dotsm,x_n$是真实序列；$Q_1,Q_2,\dotsm,Q_n$是真实值带有均值为 $0$ 噪声的观测值。就存在

$$
\begin{aligned}
    E [mean(Q_i)] &= mean(x_i) \\
    E [max(Q_i)] &\ge max(x_i) \\
    E [min(Q_i)] &\le min(x_i) \\
\end{aligned}
$$

**预测值高估：** TD目标为 $y_t = r_t + \gamma \max\limits_a Q(s_{t+1},a;w)$ ，根据上述定理可知，$\max\limits_a Q(s_{t+1},a;w)$ 的值相对于真实值，其实被放大了，进一步导致 $y_t$ 的值被高估。由于 $y_t$ 是目标，这就使得DQN模型的预测值被高估。


<span style="font-size:24px;font-weight:bold" class="section2">2. 自举</span>

在TD目标中 $y_t = r_t + \gamma \max\limits_a Q(s_{t+1},a;w)$，$Q(s_{t+1},a;w)$ 其实已经被模型高估计算了，这个高估的值又被回代用来更新 $Q(s_t,a;w)$ 的值，这就使得高估值被迭代放大。

## 高估的危害

由于DQN对于 $Q(s,a;w)$ 的高估并非均匀的，这就导致最后的预测结果大小关系的改变。这就导致「动作」的选择，可能是错误的。

<p style="text-align:center;"><img src="../../image/reinforceLearning/overestimation.jpg" width="75%" align="middle" /></p>


## 目标网络

**算法：**
1. 观测一次状态：$(s_t,a_t,r_t,s_{t+1})$
2. TD目标：$y_t = r_t + \gamma \max\limits_a Q (s_{t+1},a;w^-)$
3. TD误差：$\delta_t = Q(s_t,a_t;w) - y_t$
4. 损失函数：$L=\frac{1}{2} \delta_t^2$
5. 更新 $Q(s_t,a_t;w)$ 系数：$w = w - \alpha \delta_t \frac{\partial Q(s,a;w)}{\partial w}$
6. 更新 $Q(s,a;w^-)$ 系数：<span style="color:red;font-weight:bold"> 要一段时间才更新一次 </span>
    - **方法一：** $w^- = w$
    - **方法二：** $w^- = \tau w + (1 - \tau) w^-$

> [!note]
> 「目标网络`targe network`」方法是利用「目标网络（$Q (s_{t+1},a;w^-)$）」来代替「TD目标」中预测下一动作的动作价值的网络 $Q (s_{t+1},a;w)$。该方法避免自举。

## double DQN

**算法：**
1. 观测一次状态：$(s_t,a_t,r_t,s_{t+1})$
2. 选择下一动作：$a^* = \max\limits_a Q (s_{t+1},a;w)$
3. TD目标：$y_t = r_t + \gamma Q (s_{t+1},a^*;w^-)$
4. TD误差：$\delta_t = Q(s_t,a_t;w) - y_t$
5. 损失函数：$L=\frac{1}{2} \delta_t^2$
6. 更新 $Q(s_t,a_t;w)$ 系数：$w = w - \alpha \delta_t \frac{\partial Q(s,a;w)}{\partial w}$
7. 更新 $Q(s,a;w^-)$ 系数：<span style="color:red;font-weight:bold"> 要一段时间才更新一次 </span>
    - **方法一：** $w^- = w$
    - **方法二：** $w^- = \tau w + (1 - \tau) w^-$

> [note]
> `double DQN` 算法中，对于计算TD目标，利用 $Q (s_{t+1},a;w)$提供预测动作 $a^*$，然后利用「目标网络」 $Q (s_{t+1},a^*;w^-)$ 进行计算。


# Dueling Network

## 概念

**最优动作价值：** $Q^*(s,a) =\max\limits_\pi Q_\pi(s,a)$

**最优状态价值：** $V^*(s) =\max\limits_\pi V_\pi(s)$

**优势函数（optimal advantage function）：** 描述的是最优动作 $a$ 的优势。

$$
A^*(s,a) = Q^*(s,a) - V^*(s)
$$

**定理：**

$$
V^* (s) = \max\limits_a Q^*(s,a)
$$

## 目标

将优势函数两边最大化：

$$
\begin{aligned}
\max\limits_a A^*(s,a) &= \max\limits_a Q^*(s,a) - V^*(s) \\
&= V^*(s) - V^*(s) \\
&= 0
\end{aligned}
$$

故优势函数可以变形为：

$$
Q^*(s,a)= A^*(s,a) + V^*(s) - \max\limits_a A^*(s,a)
$$ 

> [!note]
> `dueling network`就是对上式目标 $ Q^*(s,a)$ 进行近似

# 网络结构

<p style="text-align:center;"><img src="../../image/reinforceLearning/duelingNetwork.jpg" width="75%" align="middle" /></p>

1. 对 $A^*(s,a)$ 与 $V^*(s)$ 分别采用神经网络进行近似
    - `Dense1`：近似 $A^*(s,a)$，即 $A(s,a;w^A)$
    - `Dense2`：近似 $V^*(s)$，即 $V(s;w^V)$
2. 将系数合并：$w = [w^A,w^V]$
3. 目标就为：
    $$
    Q(s,a;w)= A(s,a;w^A) + V(s;w^V) - \max\limits_a A(s,a;w^A)
    $$
4. 系数更新方法与 DQN 一样

> [!tip]
> $ mean_a A(s,a;w^A)$ 的效果可能比 $ max_a A(s,a;w^A)$ 好