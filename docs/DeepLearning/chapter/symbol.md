
# 符号约定

1. 单样本表示

    $x$为$n$维的输入；$y$为[是|否]。

    $$
    (x,y) \qquad x \in R^{n},y \in {0,1} \tag{1}
    $$

    $x$为$n_x$维的输入；$y$为$n_y$维度输出。

    $$
    (x,y) \qquad x \in R^{n_x},y \in R^{n_y} \tag{2}
    $$

1. 多样本表示

    $m$个$n_x$维的$X$

    $$
    X = \begin{bmatrix}
        \big | & \big | & \dotsm & \big | \\
        x^{(1)}  & x^{(2)} & \dotsm & x^{(m)} \\
        \big | & \big | & \dotsm & \big | \\
    \end{bmatrix} \tag{3}
    $$

    $m$个$n_y$维的$Y$

    $$
    Y = \left [ y^{(1)} \quad y^{(2)} \quad \dotsm \quad y^{(m)} \right ] \tag{4}
    $$

1. 上标$(i)$

    与第$i$个样本有关： $ z^{(i)} = w^T x^{(i)} + b$

1. 上标$[i]$

    与第$i$层神经元有关：$w^{[i]},b^{[i]}$

1. 上标$\{i\}$
    
    某个集合的子集：$x=[x_1,x_2,\dotsm,x_m],x^{\{1\}} = [x_1,x_2]$

1. 导数 $\mathrm{d} var$

    表示目标$J$关于变量$var$的导数

    $$
    \mathrm{d} var = \frac{\mathrm{d} J}{\mathrm{d} var} \tag{5}
    $$

1. i层的输出l

    a: active；表示激活函数的输出。
    $$
    a^{[i]} \tag{6}
    $$

1. 卷积神经网络

    滤波器纬度：$f^{[l]}$
    填充数：$p^{[l]}$
    步长：$s^{[l]}$
    滤波器的数量：$n_c^{[l]}$
    输出：$n_H^{[l]} \times n_W^{[l]} \times n_c^{[l]}$
    当前滤波器的规格：$f^{[l]} \times f^{[l]} \times n_c^{[l-1]}$，<span style="color:red;font-weight:bold"> 当前滤波器的深度就是上一层的滤波器个数 </span>
    
