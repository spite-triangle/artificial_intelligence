# vllm

- [vllm](https://docs.vllm.com.cn/en/latest/)

# 特点

相比于 `llama.cpp`，`vllm` 在兼顾性能的同时并发调度上优化更好
- `llama.cpp`: 个人使用
- `vllm`: 团队使用，且对运行环境要求较为苛刻，资源不够不让运行

# 安装

>[!note]
> - 没有 `windows` 版，需要使用 `wsl2`、`linux` 等系统或者 `docker`
> - `CUDA` 版本必须安装 `12` 的版本，不能 `13`，`vllm` 不支持

1. 安装 `CUDA`: 根据[官方手册](https://developer.nvidia.com/cuda-downloads?target_os=Linux)进行安装，**有专门的 `wsl2` 版本**

2. 配置环境变量

```term
triangle@LEARN:~$ vim ~/.bashrc

# 在配置文件最下面添加下列代码 12.8 对应下载的版本号
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-12.8/lib64
export PATH=$PATH:/usr/local/cuda-12.8/bin
export CUDA_HOME=$CUDA_HOME:/usr/local/cuda-12.8
export PATH=/usr/local/cuda/bin:$PATH

triangle@LEARN:~$ source ~/.bashrc
triangle@LEARN:~$  nvcc -V  // 检测配置是否成功
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2026 NVIDIA Corporation
Built on Thu_Mar_19_11:12:51_PM_PDT_2026
Cuda compilation tools, release 12.8, V12.8.78
Build cuda_12.8.r12.8/compiler.37668154_0
```

3. 安装 `vllm`

> [!note]
> `python` 版本最好 `3.12` 兼容性好

```term
triangle@LEARN:~$ pip isntall uv
triangle@LEARN:~$ uv venv --seed --python 3.12 // 在目标目录生成 vllm 的安装环境
triangle@LEARN:~$ uv pip install vllm --torch-backend=auto
```

4. [可选]设置 `Hugging Face` 模型下载目录与国内镜像

```term
triangle@LEARN:~$ vim ~/.bashrc

export HF_HOME=/data/my_huggingface_cache
export HF_ENDPOINT=https://hf-mirror.com
triangle@LEARN:~$ source ~/.bashrc
```


