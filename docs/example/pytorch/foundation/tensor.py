# %% import
import torch

# %% 创建 Tensor

# 直接创建 Tensor 类
mat = torch.Tensor([[1, 2], [3, 3]])

# 通过函数创建
mat = torch.tensor([[1, 2], [3, 3]])


# 单位阵
matE = torch.eye(3, 5)

# 创建全1.全0
matOne = torch.ones(2, 3)
matZore = torch.zeros(2, 3)

# 随机创建
mat = torch.rand(6, 6)
mat = torch.randn(5, 4)

# 不同位置的元素，指定不同的正态分布
mat = torch.normal(mean=torch.ones((3, 4)), std=10*torch.rand((3, 4)))

# 根据现有的矩阵的形状，进行创建
matOne = torch.ones_like(mat)
matZore = torch.zeros_like(mat)
mat = torch.randn_like(mat)
mat = torch.rand_like(mat)

# 生成序列
vec = torch.arange(0, 15, 1)
vec = torch.linspace(0, 15, 16)

# 生成随机索引序列
vec = torch.randperm(16)

# 维度索引
index = torch.tensor([[1, 3, 6], [2, 4, 6]])
# 值
value = torch.tensor([1, 2, 3])
# 稀疏矩阵
matSpares = torch.sparse_coo_tensor(index, value, (8, 7))
# 将稀疏矩阵转为稠密形式
mat = matSpares.to_dense()


# %% 属性
mat = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.float,
                   device=torch.device('cuda'))

# 形状
mat.shape

# 数据类型
mat.dtype

# 存储位置
mat.device

# %% 运算
a = torch.tensor([[1, 2, 3], [3, 4, 5], [6, 7, 8]])
b = torch.tensor([[3, 4, 5], [1, 2, 3], [6, 7, 8]])

# 四则
c = a + b
c = a / b
c = a - b
c = a * b

# 矩阵
c = a @ b
c = torch.matmul(a, b)
c = torch.mm(a,b)

# 张量
a = torch.rand(2,4,3)
b = torch.rand(2,3,5)
c = torch.matmul(a, b)
print(c.shape)

# %% 广播

a = torch.rand(2,3,1)
b = torch.rand(4)
c = a + b
print(a,b,c)

# %% 比较

a = torch.tensor([[1,2,3],[4,1,6]],dtype=torch.float)
b = torch.tensor([[1,2,3],[1,3,6]],dtype=torch.float)

# 对应元素比较，返回结果为张量
a > b
a < b
a == b

# 对比两个张量是否完全一样：维度，数值均相同。返回一个 bool 值
torch.equal(a, b)

# %% 排序

a = torch.tensor([[5,2,7,6],[4,1,6,8]],dtype=torch.float)

# 根据 dim 维度方向，对元素进行排序
values,indices = torch.sort(a,dim=1,descending=False)

# 根据 dim 维度方向，对元素进行排序。然后输出前 k 个值
values,indices = torch.topk(a,k=1,dim=1) 

# 根据 dim 维度方向，对元素进行排序。然后输出排在第 k 个位置的值
values,indices = torch.kthvalue(a, k=2,dim=0)

# %% 筛选
a:torch.Tensor = torch.tensor([[5,2,7,6],[4,1,6,8]],dtype=torch.float)
b = torch.tensor([[1,2,3,6],[2,3,3,8]],dtype=torch.float)

# 筛选出所有小于 3 的值，并修改
a[a<3] =  1

# 遮罩选择
vec = torch.masked_select(a, a>3)

# 降成1维，然后进行选择
vec = torch.take(a,index=torch.tensor([2,7]))

# 获取 a 大于 1 的值的索引
indices=torch.where(a>1)

# a > 1 的位置填入 a 的元素；
# a <= 1 的位置填入 b 的元素
mat = torch.where(a>1,a,b)

# 选出索引为 index 的 dim 维度。下面操作同 a[index,:] 
mat = torch.index_select(a, dim=0, index=torch.tensor([0]))

# 在 dim 方向上，选择 index 对应的元素：下面操作同 a[i,index[i,j]]
mat = torch.gather(a, dim=1, index=torch.tensor([ [2,3],[1,2] ]))

torch.nonzero(a)

# %% 变形
a:torch.Tensor = torch.tensor([[5,2,7,6],[4,1,6,8]],dtype=torch.float)
b = torch.tensor([[1,2,3,6],[2,3,3,8]],dtype=torch.float)

# 改变维度
a.reshape(2,2,2)

# 在 dim 方向上，将元素拼接起来，拼接后维度不变
torch.cat([a,b],dim=0)
torch.vstack(a,b)
torch.hstack(a,b)

# 在 dim 方向上，将张量拼接起来，拼接后维度加一
torch.stack([a,b],dim=2)

torch.split()