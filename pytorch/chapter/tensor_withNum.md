# Tensor

# 1. 介绍


**Tensor**：张量， PyTorch 中的基本运算的数据类型，**实际上就是一个多维数组，并且可以看作是 Numpy 的翻版实现**。

- <a href="https://www.bilibili.com/video/BV1Ja4y1e7bu" class="jump_link"> 张量的数学解释 </a> ：感兴趣了解一下

> [!tip|style:flat]
> Tensor 可以直接当做 Numpy 使用，API 接口基本一致。




# 2. 创建

## 2.1. 常规创建

```python
import torch

# 直接创建 Tensor 类
mat = torch.Tensor([[1,2],[3,3]])

# 通过函数创建
mat = torch.tensor([[1,2],[3,3]])

# 单位阵
matE = torch.eye(3,5)

# 创建全1.全0
matOne = torch.ones(2,3)
matZore = torch.zeros(2,3)

# 随机创建
mat = torch.rand(6,6)
mat = torch.randn(5,4)

# 不同位置的元素，指定不同的正态分布
mat = torch.normal(mean=torch.ones((3,4)), std=10*torch.rand((3,4)))

# 根据现有的矩阵的形状，进行创建
matOne = torch.ones_like(mat)
matZore = torch.zeros_like(mat)
mat = torch.randn_like(mat)
mat = torch.rand_like(mat)

# 生成序列
vec = torch.arange(0, 15, 1)
vec = torch.linspace(0, 15,16)

# 生成随机索引序列
vec = torch.randperm(16)
```

## 2.2. 稀疏矩阵

- **稀疏矩阵：** 矩阵中大量含有 0
- **稠密矩阵：** 矩阵中几乎没有 0

```python
# 维度索引
index = torch.tensor([[1,3,6],[2,4,6]])
# 值
value = torch.tensor([1,2,3])
# 稀疏矩阵
matSpares = torch.sparse_coo_tensor(index, value,(8,7))
# 将稀疏矩阵转为稠密形式
mat = matSpares.to_dense()
```

# 3. 属性

```python
mat = torch.tensor([[1,2,3],[4,5,6]],dtype=torch.float,device=torch.device('cuda'))

# 形状
mat.shape

# 数据类型
mat.dtype

# 存储位置
mat.device

# 矩阵形式：稀疏还是稠密
mat.layout

# 真正的数据
mat.data
```

> [!note]
> 利用 `mat` 进行运算时，会导致数据地址的改变，例如 `mat= mat - 1`（Python 中运算前和运算后的 `mat` 就是两个不同的实列了），这会可能出问题（例如，之后章节中的计算图），因此对于数据的更新操作，最好对`mat.data`进行操作。

# 4. 运算

## 4.1. 四则运算

**与 Numpy 一样，对应元素相加减，乘除**

```python
a = torch.tensor([[1, 2, 3], [3, 4, 5], [6, 7, 8]])
b = torch.tensor([[3, 4, 5], [1, 2, 3], [6, 7, 8]])

c = a + b
c = a / b
c = a - b
c = a * b
```

## 4.2. 张量乘法

- **二维矩阵乘法：**
    ```python
    # 二维矩阵
    a = torch.tensor([[1, 2, 3], [3, 4, 5], [6, 7, 8]])
    b = torch.tensor([[3, 4, 5], [1, 2, 3], [6, 7, 8]])

    # 矩阵运算
    c = a @ b
    c = torch.mm(a,b)
    c = torch.matmul(a, b)
    ```
- **高维张量乘法：** 
  - 前 `n-2` 维度相同
  - 后 `n-1,n` 做矩阵乘法
  假设张量 A 为 `(2,3,4,5)`，张量 B 为 `(2,3,5,6)`。两张量，前两个维度一样，均为`2,3`，后两个两个维度可以做矩阵乘法 `(4,5)x(5,6)`，张量的乘法计算结果就为 `(2,3,4,6)`，**即两个高维数组中的矩阵对应相乘。**

   ```python
    a = torch.rand(2,4,3)
    b = torch.rand(2,3,5)

    # 只能用这个，@ mm 只能计算矩阵的
    c = torch.matmul(a, b)
   ```

## 4.3. 范数

**范数：** 一个矩阵的大小，比较常用的就是向量的模。

```python
# 计算 C=(A - B) 的范数
# p：指定几范数
torch.dist(A,B,p)

# 计算一个矩阵的二范数
torch.norm(A)
```


## 4.4. 判断

```python
a = torch.tensor([[1,2,3],[4,5,6]])
b = torch.tensor([[1,2,3],[1,3,6]])

# 对应元素比较，返回结果为张量
a > b
a < b
a == b

# 判断元素是否为 nan 值
torch.isnan(a)
# 判断元素是否为 无限值
torch.isinf(a)

# 对比两个张量是否完全一样：维度，数值均相同。返回一个 bool 值
torch.equal(a, b)
```

## 4.5. clamp

**作用：** 将 tensor 值限制在某个区间内

```python
torch.clamp(tensor,min,max)
```

## 4.6. in-place

**作用：** 直接在 tensor 进行修改，不产生中间结果。**所有运算函数名末尾带`_`的都是。**

```python
A = torch.rand(3,4)
B = torch.rand(3,4)

A.add_(B)
A.pow_(2)
A.clamp_(1,2)
...
```

## 4.7. 广播机制

- **作用**： 当两个张量不能满足上述运算的维度关系时，对于特定条件的维度，可以根据「广播机制」进行维度扩充，进而实现对应计算。

- **广播条件：**
    1. 每个张量至少要有一个维度
    2. 维度右对齐，且对应后维度要么其中一个没有，要么其中一个为 1。
- **广播案例：** 假设张量 A 维度为 `(2,3,1)`与张量 B 维度为 `(4)`，计算`A + B`
    
    首先维度右对齐
    
    $$
    \begin{aligned}
        A: (2,3,1) & \\    
        B: \qquad  (4) & \\    
    \end{aligned}
    $$

    `1` 与 `4` 其中一个维度为 `1`，因此对 A 的 `1` 维度进行复制

    $$
    \begin{aligned}
        A: (2,3,4) & \\    
        B: \qquad  (4) & \\    
    \end{aligned}
    $$

    又由于 B 缺少两个维度，因此对 B 复制出两个维度

    $$
    \begin{aligned}
        A: (2,3,4) & \\    
        B: (2,3,4) & \\    
    \end{aligned}
    $$

    最后按照运算规则进行计算

    ```python
    a = torch.rand(2,3,1)
    b = torch.rand(4)
    c = a + b
    ```

## 4.8. 其他

最大值、取整、三角函数、对数、绝对值等，都和 Numpy 一样。

```python
torch.sqrt()
torch.sin()
torch.argmax(A,dim)
torch.sum(A,dim)
....
```

# 5. 筛选

## 5.1. dim

**含义：** 张量维度的编号，从 0 开始。
    $$
    \begin{aligned}
        dim: \quad 0,1,2,3 & \\
        shape:(10,3,4,5 &)  \\
    \end{aligned}
    $$

在函数中，设定维度编号，就表示对哪个维度进行操作。

```python
A = torch.tensor([[1,2],[3,4]])

# 对矩阵进行列相加。也就是对第 1 维度进行加和操作
torch.sum(A,dim=1)

# 对矩阵进行行相加。也就是对第 0 维度进行加和操作
torch.sum(A,dim=0)
```

## 5.2. 排序筛选

```python
a = torch.tensor([[5,2,7,6],[4,1,6,8]],dtype=torch.float)

# 根据 dim 维度方向，对元素进行排序
# values：排序后的结果，为 Tensor
# indices：values 的元素对应原 a 的索引
values,indices = torch.sort(a,dim=0,descending=False)

# 根据 dim 维度方向，对元素进行排序。然后输出前 k 个值
values,indices = torch.topk(a,k=1,dim=1) 

# 根据 dim 维度方向，对元素进行排序。然后输出排在第 k 个位置的值
values,indices = torch.kthvalue(a, k=2,dim=0)
```

## 5.3. 条件筛选


```python
# %% 筛选
a = torch.tensor([[5,2,7,6],[4,1,6,8]],dtype=torch.float)
b = torch.tensor([[1,2,3,6],[2,3,3,8]],dtype=torch.float)

# 筛选出所有小于 3 的值，并修改
a[a<3] =  1

# 遮罩选择，下面操作同：a[a>3]
vec = torch.masked_select(a, a>3)

# 降成1维，然后进行选择：torch.flatten(a)[index]
vec = torch.take(a,index=torch.tensor([2,7]))

# 非零
indices = torch.nonzero(a)

# 获取 a 大于 1 的值的索引
indices = torch.where(a>1)

# a > 1 的位置填入 a 的元素；
# a <= 1 的位置填入 b 的元素
mat = torch.where(a>1,a,b)

# 选出索引为 index 的 dim 维度。下面操作同 a[index,:] 
mat = torch.index_select(a, dim=0, index=torch.tensor([0]))

# 在 dim 方向上，选择 index 对应的元素：下面操作同 a[i,index[i,j]]
mat = torch.gather(a, dim=1, index=torch.tensor([ [2,3],[1,2] ]))
```
## 5.4. 索引筛选

- 在索引中写入数组：根据传入的数组拿出数据，然后根据拿出的顺序组合成新的数组

    <!-- panels:start -->
    <!-- div:left-panel -->
    ```python
    a = torch.rand((3,3,2))

    # 两种写法等价
    a[[0,1,0,0],:,:].shape
    a[[0,1,0,0]].shape    
    ```
    
    <!-- div:right-panel -->

    ```term
    triangle@LEARN:~$ python ./test.py
    torch.Size([4, 3, 2])
    torch.Size([4, 3, 2])
    ```
    <!-- panels:end -->

- `bool` 遮罩：利用一个 `bool` 数组对目标 Tensor 进行标记

    <!-- panels:start -->
    <!-- div:left-panel -->
   ```python
    a = torch.rand((3,3,2))

    # 与 a 同等 shape 的 bool 标记
    mask = [[[False,  True],
            [False, False],
            [False,  True]],

            [[False,  True],
            [False,  True],
            [False, False]],

            [[False,  True],
            [False, False],
            [False, False]]]
    a[mask].shape

    # 对某一维度进行标记，bool 数组要和该维度的长度相等
    mask = [True,False,True]
    a[:,mask].shape
    ```
    
    <!-- div:right-panel -->
    ```term
    triangle@LEARN:~$ python ./test.py
    torch.Size([5])
    torch.Size([3, 2, 2])
    ```
    <!-- panels:end -->

- 特殊符号
    <!-- panels:start -->
    <!-- div:left-panel -->
    ```python
    a = torch.rand((3,3,2))

    # : 表示对应维度的所有元素
    a[:,:,0].shape
    # ... 维度要从后向前看
    a[...,1].shape
    # 生成一个数组，等同于 [0,1] 
    a[0:2].shape
    ```
    
    <!-- div:right-panel -->
    ```term
    triangle@LEARN:~$ python ./test.py
    torch.Size([3, 3])
    torch.Size([3, 3])
    torch.Size([2, 3, 2])
    ```    
    <!-- panels:end -->

- 索引结果的维度变化


    <!-- panels:start -->
    <!-- div:left-panel -->
    
    ```python
    a = torch.rand((3,3,2))

    # 索引中存在常值，就为降维，有多少个常值，就降低几维
    a[0,:,0].shape # 降 2 维
    a[:,0].shape # 降 1 维

    # 索引为数组时，维度不变
    a[0].shape
    a[[0]].shape # 和下面写法等价
    a[0:1].shape
    ```    
    <!-- div:right-panel -->
    ```term
    triangle@LEARN:~$ python ./test.py
    torch.Size([3])
    torch.Size([3, 2])
    torch.Size([3, 2])
    torch.Size([1, 3, 2])
    torch.Size([1, 3, 2])
    ```
    <!-- panels:end -->
# 6. 变形

## 6.1. 组合拼接

```python
a = torch.tensor([[5,2,7,6],[4,1,6,8]],dtype=torch.float)
b = torch.tensor([[1,2,3,6],[2,3,3,8]],dtype=torch.float)

# 改变维度
a.reshape(2,2,2)

# 在 dim 方向上，将元素拼接起来，拼接后维度不变
torch.cat([a,b],dim=0)
torch.vstack(a,b)
torch.hstack(a,b)

# 在 dim 方向上，将张量拼接起来，拼接后维度加一
torch.stack([a,b],dim=2)
```

## 6.2. 切片

```python
# 在dim维度上，平均拆分成 chunks 块
torch.chunk(tensor,chunks,dim)

# 在dim维度上，指定 split_size_or_sections 的大小为一块
torch.split(tensor,split_size_or_sections,dim)
```

