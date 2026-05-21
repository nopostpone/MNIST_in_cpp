# MNIST CNN in C++

用纯 C++ (Eigen, 无深度学习框架) 实现的 MNIST 手写数字识别 CNN，97.6% 测试准确率。

## 快速开始

```bash
# 1. 编译（需要 CMake + g++，自动下载 Eigen 3.4.0）
cd build && cmake -DCMAKE_BUILD_TYPE=Release .. && mingw32-make

# 2. 训练（生成 model_weights.bin）
./build/mnist_cpp.exe train

# 3. 启动 Web 验证
cd web && python server.py
# 浏览器打开 http://localhost:5000，在画布上写数字验证
```

需要 Python 依赖：`pip install flask`

## 架构

```
Input (1×28×28)              [1, 28, 28]
    ↓
Conv2D(1→32, k=3, p=1)     → 32×28×28   参数量: 320
    ↓ ReLU
    ↓ MaxPool2D(2×2)        → 32×14×14
    ↓
Conv2D(32→64, k=3, p=0)    → 64×12×12   参数量: 18,496
    ↓ ReLU
    ↓ MaxPool2D(2×2)        → 64×6×6
    ↓
Flatten                     → 2304
    ↓
Linear(2304→128)            → 128        参数量: 294,912
    ↓ ReLU
    ↓
Linear(128→10)              → 10         参数量: 1,290
────────────────────────────────────────
总参数: ~315,000
```

CrossEntropyLoss 内部计算 softmax，模型最后一层不加 Softmax。

## 训练结果

| Epoch | Train Loss | Train Acc | Test Acc | 时间 |
|-------|-----------|-----------|----------|------|
| 1 | 0.2756 | 91.76% | 95.23% | ~72s |
| 2 | 0.1041 | 96.86% | 96.29% | ~72s |
| 3 | 0.0698 | 97.90% | 97.05% | ~72s |
| 4 | 0.0495 | 98.56% | 97.43% | ~72s |
| 5 | 0.0354 | 99.01% | 97.63% | ~72s |

Batch size=64, lr=0.001, O3 优化, ~585 samples/s.

## 项目结构

```
src/
├── main.cpp                  # 入口（train / predict 模式）
├── core/module.h             # Parameter, Module 基类, 权重序列化
├── data/mnist_loader.*       # MNIST IDX 格式解析
├── layers/
│   ├── conv2d.*              # 2D 卷积 (im2col + GEMM, 批量)
│   ├── max_pool.*            # 2D 最大池化 (批量)
│   ├── flatten.*             # 展平层
│   └── fully_connected.*     # 全连接层 (批量)
├── activations/
│   ├── relu.*                # ReLU 激活
│   └── softmax.*             # Softmax (未使用，由 CrossEntropyLoss 替代)
├── loss/cross_entropy.*      # 交叉熵损失 (内置 softmax)
├── model/
│   ├── model.*               # Sequential 容器
│   └── trainer.*             # 训练循环 (小批量 SGD)
└── optimizer/sgd.*           # SGD 优化器 (未编译)
web/
├── server.py                 # Flask Web 服务
└── templates/index.html      # Canvas 画布前端
```

## 各层数学原理

### Conv2D — im2col + GEMM

将卷积转化为矩阵乘法。对每个输出位置，提取输入中 `K×K` 的感受野，展开为列向量。所有列拼接为 `col` 矩阵，然后：

```
前向:  out = W · col + b        W: (out_c, in_c·K·K)   col: (in_c·K·K, H_out·W_out)
反向:  dW += dout · col^T
       db += sum(dout, axis=1)
       dx = col2im(W^T · dout)
```

批量：拼接 B 个样本的 im2col 结果，一次大矩阵乘法。

### MaxPool2D

```
前向:  out(ch, i, j) = max_{pi, pj}(input(ch, i·S+pi, j·S+pj))
       记录最大值位置 mask
反向:  dx(mask) += dout    (梯度只流向最大值位置)
```

### CrossEntropyLoss

```
前向:  L = -log(softmax(logits)[label])
反向:  dL/d(logits) = softmax(logits) - one_hot(label)
```

内置 softmax，直接接收 logits。

## 命令行用法

```bash
# 训练
./build/mnist_cpp.exe train

# 推理（从 stdin 读取 784 字节的原始灰度值）
echo ... | ./build/mnist_cpp.exe predict
```

## 权重文件格式

`model_weights.bin` 为自定义二进制格式：

```
[4B] 参数个数 n
重复 n 次:
  [4B] rows  [4B] cols  [rows*cols*4B] float data
```

## 已知问题

- GCC 15.2.0 (MSYS2/MinGW) 的 `std::ifstream`/`std::ofstream` 在 O1+ 崩溃。`mnist_loader.cpp` 和权重 I/O 已改用 C FILE* 绕过。
- OpenMP 对此模型规模加速有限，保持关闭。
- Eigen 未链接 BLAS，矩阵乘速度为纯 C++ 实现。
