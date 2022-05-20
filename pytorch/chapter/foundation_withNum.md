# 基本概念

# 1. 介绍

<p style="text-align:center;"><img src="/artificial_intelligence/image/pytorch/pytorch-logo.png" width="25%" align="middle" /></p>

Pytorch 是 torch 的 python 版本，是由 Facebook 开源的神经网络框架，专门针对 GPU 加速的深度神经网络（DNN）编程。Torch 是一个经典的对多维矩阵数据进行操作的张量（tensor）库，在机器学习和其他数学密集型应用有广泛应用。与 Tensorflow1.0 的静态计算图不同，Pytorch 的计算图是动态的，可以根据计算需要实时改变计算图。


# 2. 安装

- <a href="https://pytorch.org/get-started/locally/" class="jump_link"> PyTorch 安装 </a>

```python
import torch
# 检测 CUDA 是否安装成功
print(torch.cuda.is_available())
```

# 3. 基本结构

<p style="text-align:center;"><img src="/artificial_intelligence/image/pytorch/baseStructure.png" width="50%" align="middle" /></p>

对于 PyTorch 的结构组成主要分为三个部分：
- **Tensor**：张量， PyTorch 中的基本运算的数据类型，实际上就是一个多维数组，并且可以看作是 Numpy 的翻版实现。
- **Variable**：变量，能进行梯度求解（Autograd）的数据类型。在新版本中，Variable 与 Tensor 数据类型进行了合并。
- **nn.Module**：神经网络模型