# Actor-Critic算法

# 1. 算法概念

**思路：** 在 [策略学习](./ReinforcementLearning/chapter/policybasedLearning_withNum.md) 对 $Q_\pi (s_t,a_t)$ 的计算是用一次回合后得到的 $U$ 进行近似。同样对于 $Q_\pi (s_t,a_t)$ 也可以再建立一个神经网络进行拟合。

**策略网络 $\pi (a|s;\theta)$ ：**  用于对策略函数的近似，即`Actor`，控制角色的动作

**价值网络 $q (s,a;w)$ ：** 用于对价值函数的近似，即`Critic`，对角色动作进行打分

**总目标：** 使得最终的状态价值函数的值最大，$V(s;\theta,w) = \sum\limits_a \pi (a|s;\theta) q (s,a;w)$

# 2. 网络模型

**策略网络：**

<p style="text-align:center;"><img src="../../image/reinforceLearning/example_policy_idea.jpg" width="75%" align="middle" /></p>

1. 一张游戏画面当作一个「状态」，状态通过「卷积层」实现特征提取
2. 特征输入给「权连接层」进行 $\pi(a|s)$ 拟合
3. 将「权连接层」的输出通过`softmax`层转化为概率
4. 输出每个动作在当前状态下产生好结果的概率

**价值网络**

<p style="text-align:center;"><img src="../../image/reinforceLearning/example_critic.jpg" width="75%" align="middle" /></p>

1. 一张游戏画面当作一个「状态」，状态通过「卷积层」实现特征提取
2. 要评价的一个「动作」通过权连接层进行特征提取
3. 将「状态」与「动作」特征进行组合，最后输入一个权连接层，得到「评分」


# 3. 网络训练

**目标：**
- **策略网络：** 价值函数 $V(s)$ 的值最大化
- **价值网络：** 对动作的打分 $q(s,a)$ 更接近真实值

**参数更新：**
1. 观测到环境状态 $s_t$
2. 根据 $\pi(a|s;\theta)$ 得到动作 $a_t$
3. 角色执行动作 $a_t$，然后获取状态 $s_{t+1}$ 与奖励 $r_t$
4. 根据[TD算法](./ReinforcementLearning/chapter/valuebasedLearning_withNum.md)更新系数 $w$
5. 根据[策略梯度算法](./ReinforcementLearning/chapter/policybasedLearning_withNum.md)更新系数 $\theta$

**系数 $w$ 更新：**
1. 计算 $q(s_t,a_t;w_t)$
2. 根据状态 $s_{t+1}$，通过 $\pi(a|s;\theta)$ <span style="color:red;font-weight:bold"> 预测动作 $a_t$ </span> ，然后计算 $q(s_{t+1},a_{t+1};w_t)$
3. 计算TD目标：$y_t = r_t + \gamma q(s_{t+1},a_{t+1};w_t)$
4. 损失函数：$L(w) = \frac{1}{2} [q(s_t,a_t;w_t) - y_t]^2$ 
5. 更新参数：$w_{t+1} = w_{t} - \alpha \frac{\partial L(w)}{\partial w}$

**系数 $\theta$ 更新：**
1. 根据策略函数 $\pi (a|s_t;\theta_t)$ 随机获取动作 $\hat a$
2. 带入 $\hat a$ 计算：$g(\hat a,\theta_t) = \frac{\partial \ln \pi(\hat a|s;\theta_t)}{\partial \theta} q (s_t,\hat a;w)$
3. 然后通过 $g(\hat a,\theta_t)$ 来近似计算：$\frac{\partial V(s;\theta_t)}{\partial \theta_t} =E_A[g(A,\theta_t)] $
4. 更新参数：$\theta_{t+1} = \theta_t + \beta g(a,\theta_t)$

> [!note]
> 再计算 $g(a,\theta)$ 时，一般会使用TD error：
> $$\delta_t = q(s_t,a_t;w_t) - [r_t + \gamma q(s_{t+1},a_{t+1};w_t)]$$
> 来代替 $q (s_t,a;w)$，这样可以使算法收敛加快。