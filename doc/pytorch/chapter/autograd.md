# Autograd

- <a href="https://www.bilibili.com/video/BV1LL41147G8" class="jump_link"> PyTorch 自动梯度 </a>
- <a href="https://spite-triangle.github.io/artificial_intelligence/#/./DeepLearning/chapter/foundation" class="jump_link"> 神经网络反向传播 </a>

# Variable

**定义**：在 PyTorch 中，对于 Variable 的定义就是 **可以用于求导的 Tensor**。

将 `requires_grad` 设置为 `True` 就可以定义一个变量了

```python
# 构建 变量
x = torch.tensor([2],requires_grad=True) 

# 变量的微分计算式子，该属性有值才能参与微分计算
x.grad_fn
```
- **grad_fn 自动生成：** 利用具有 `grad_fn` 属性的张量参与数学运算后，计算得到的张量结果也会具有 `grad_fn`。例如 x 具有 grad_fn ，通过 `y=x**2+1`得到的新张量 y 也会具有 grad_fn 属性。

# 动态计算图

**定义：** 节点表示张量，边表示张量之间运算逻辑。
- 计算图的正向传播是立即执行
- 计算图在反向传播后立即销毁

<p style="text-align:center;"><img src="../../image/pytorch/dynamicGraph.gif" width="50%" align="middle" /></p>

以一个简单的神经元为例：

```python
x = torch.tensor([[2],[3]],dtype=torch.float)

# 变量
w = torch.tensor([[1,2]],requires_grad=True,dtype=torch.float) 
b = torch.tensor([1],requires_grad=True,dtype=torch.float) 

# 权值计算
z = torch.mm(w,x) + b

# 激活函数
a = g(z)
```

根据上述计算流程，PyTorch 就能自动搭建一个动态计算图

<p style="text-align:center;"><img src="../../image/pytorch/cellCompute.png" width="50%" align="middle" /></p>

- 叶子结点：计算图的输入，由用户直接定义，不是靠函数关系计算得到
- 中间结点：通过函数关系，计算得到的中间变量
- 输出结点：最后的输出结果

**对 tensor 进行叶子结点判断时，`x` 也会认为是叶子结点，虽然 `x` 不能用于梯度求解**

```python
# 是否为叶子结点
a.is_leaf
```

# 反向传播

## 梯度求解

同样以神经元为例，为了方便计算，假设激活函数为

$$g(x)=x^2$$

则「前向传播」计算流程就为

$$
\begin{aligned}
    z &= wx + b \\
    a &= z^2
\end{aligned}
$$

```python
x = torch.tensor([[2],[3]],dtype=torch.float)

# 变量
w = torch.tensor([[1,2]],requires_grad=True,dtype=torch.float) 
b = torch.tensor([1],requires_grad=True,dtype=torch.float) 

# 权值计算
z = torch.mm(w,x) + b

# 激活函数
a = z**2
```
从 a 开始执行「反向传播」

$$
\begin{aligned}
    \frac{d a}{d z} &= 2z \\
    \frac{d z}{d w} &= x^T \\
    \frac{d z}{d b} &= 1 \\
\end{aligned}
$$

根据链式求导可知

$$
\begin{aligned}
    \frac{d a}{d w} &= 2zx^T \\
    \frac{d a}{d b} &= 2z \\
\end{aligned}
$$

```python
# 从 a 开始反向传播
a.backward()
```

获取系数 $w,b$ 的梯度。<span style="color:red;font-weight:bold"> 默认情况下，PyTorch 不会保留「中间结点」的梯度，即 $z$ 的梯度值为 `None` </span>

<!-- panels:start -->
<!-- div:left-panel -->

```python
print(w.grad)
print(b.grad)
print(z.grad)
print(x.grad)
```
<!-- div:right-panel -->

```term
triangle@LEARN:~$ python autograd.py
tensor([[36., 54.]])
tensor([18.])
None
None
```
<!-- panels:end -->

**由于我们是从 a 开始进行反向传播的**，因此 `w.grad` 对应的值就是
$$
\begin{aligned}
z&=wx+b=9 \\
\frac{da}{dw} |_{w=[1,2],b=1,x=[2,3]^T} &= 2zx|_{w=[1,2],b=1,x=[2,3]^T} \\
&= \begin{bmatrix}
   2*9*2=36 \\
   2*9*3=54 
\end{bmatrix}
\end{aligned}
$$

其他同理。

> [!tip]
> - 计算图中，当某一张量 $\Lambda$ 进行「反向传播」，那么「叶子节点 $\Omega$」的 `grand` 属性就是 $\frac{d \Lambda}{d \Omega}$ 的值
> - 某一张量进行「反向传播」后，计算图将销毁。


## 梯度计算控制

- **中间结点梯度保存**
   ```python
   # 权值计算
    z = torch.mm(w,x) + b
    z.retain_grad()
    # 激活函数
    a = z**2
   ```

- **阻止计算图追踪**
  - 方法一
    ```python
        # 权值计算
        z = torch.mm(w,x) + b
        # 后续计算被计算图屏蔽
        with  torch.no_grad():
            # 激活函数
            a = z**2
    ```
  - 方法二
    ```python
        # 权值计算
        z = torch.mm(w,x) + b
        
        # 对 z 进行重建，z1 将不能用于求解梯度 
        z1 = z.detach()
        
        # 激活函数
        a = z1**2
    ```

## 自定义梯度计算

- **目的：** 通过自定翼运算规则，将 $z = wx + b$ 封装为一个运算类。

    前向传播计算流程

    $$
    \begin{aligned}
        z &= wx + b \\
        a &= z^2
    \end{aligned}
    $$

    从 a 开始执行反向传播

    $$
    \begin{aligned}
        \frac{d a}{d z} &= 2z \\
        \frac{d z}{d w} &= x^T \\
        \frac{d z}{d b} &= 1 \\
    \end{aligned}
    $$

    根据链式求导可知

    $$
    \begin{aligned}
        \frac{d a}{d w} &= \frac{d a}{d z} x^T \\
        \frac{d a}{d b} &= \frac{d a}{d z} \\
    \end{aligned}
    $$

- **实现：** 继承`torch.autograd.Function`类，并重写前向传播`forward`与反向传播`backward`方法


   ```python
    class line(torch.autograd.Function):
        @staticmethod
        def forward(ctx,w,b,x):
            # 保存输入值，供反向传播的梯度计算
            ctx.save_for_backward(w,b,x)
            return torch.mm(w, x) + b

        @staticmethod
        def backward(ctx,grad_output):
            """ grad_output：为链式求导法则上一级的导数结果 """

            w,b,x = ctx.saved_tensors
            dw = torch.mm(grad_output,x.t())
            db = grad_output

            # 由于不计算 x 的梯度，所以返回 None
            return dw,db,None

    x = torch.tensor([[2],[3]],dtype=torch.float)
    # 变量
    w = torch.tensor([[1,2]],requires_grad=True,dtype=torch.float) 
    b = torch.tensor([1],requires_grad=True,dtype=torch.float) 

    # 调用自定义封装
    z = line.apply(w,b,x)

    # 激活函数
    a = z**2

    # 反向传播
    a.backward()
   ```

# 梯度下降法简单应用

- <a href="https://blog.csdn.net/pupilxmk/article/details/80735599" class="jump_link"> 梯度下降法 </a> ：梯度下降法和牛顿法很相似，可以从牛顿法的角度出发理解：牛顿法的学习率是精确计算结果，而梯度下降法是瞎给的。

- **问题：** 
    $$
    \min : f(x_1,x_2) = (x_1 + 10)^2 + (x_2 - 5)^2
    $$

   ```python
    # 初始化
    x1 = torch.tensor([1],dtype=torch.float,requires_grad=True)
    x2 = torch.tensor([1],dtype=torch.float,requires_grad=True)

    # 学习率
    alpha = 0.5

    # 迭代次数
    n = 10

    for epoch in range(n):
        # 前向计算
        f = (x1-10)**2 + (x2+5)**2
        # 反向计算
        f.backward()
        
        # 修正 x1 x2
        x1.data = x1.data - alpha * x1.grad
        x2.data = x2.data - alpha * x2.grad

    print(x1,x2)
   ```

> [!note|style:flat]
> 这里更新数据要使用 `x1.data`，直接对数据进行修改。若利用 `x1` 进行运算，就会导致 `x1` 被赋予 `grad_fn`，这样下一次迭代时，`x1` 将不在是「叶子结点」而转变为「中间结点」，默认梯度将不会保留，就会导致疯狂报错。

