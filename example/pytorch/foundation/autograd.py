# %% 
import torch
# %% 变量

x = torch.tensor([[2],[3]],dtype=torch.float)

# 变量
w = torch.tensor([[1,2]],requires_grad=True,dtype=torch.float) 
b = torch.tensor([1],requires_grad=True,dtype=torch.float) 

# 权值计算
z = torch.mm(w,x) + b

# 激活函数
a = z**2

# 反向传播
a.backward()

print(a.is_leaf)
print(x.is_leaf)
print(z.is_leaf)

# 变量梯度
print(w.grad)
print(b.grad)
print(z.grad)
print(x.grad)

# %% 自定义规则

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

# %% 求解简单最小化问题

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
