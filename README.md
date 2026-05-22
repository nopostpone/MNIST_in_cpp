# MNIST CNN in C++

纯 C++ (Eigen, 无深度学习框架) 实现的 MNIST 手写数字识别 CNN，**98.4% 测试准确率**。

详细架构设计原理见 [docs/tutorial.md](docs/tutorial.md)。

## 环境要求

- **CMake** ≥ 3.16 + 支持的构建工具（自动下载 Eigen 3.4.0）
- **g++** (C++17，已在 GCC 15.2.0 测试)
- **Git**（CMake FetchContent 需要）

## 快速开始

```bash
# 1. 克隆
git clone https://github.com/nopostpone/MNIST_in_cpp.git
cd MNIST_in_cpp

# 2. 编译（自动下载 Eigen 3.4.0）
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build

# 3. 训练（~3 分钟，生成 model_weights.bin）
./build/mnist_cpp train

# 4. 验证（128 样本快速测试）
./build/mnist_cpp sanity
```

### 在线演示

打开 [GitHub Pages 演示页](https://nopostpone.github.io/MNIST_in_cpp)，直接在浏览器里手写数字，推理完全在本地 JS 引擎完成。

### 导出权重

训练后导出 JSON 给静态网页用：

```bash
./build/mnist_cpp export > docs/weights.json
```

## 架构

```
Input (1×28×28)
    ↓
Conv2D(1→10, k=3)         → 10×26×26    100 参数
    ↓ ReLU
Conv2D(10→10, k=3)        → 10×24×24    910 参数
    ↓ ReLU
MaxPool2D(2×2)            → 10×12×12
    ↓
Conv2D(10→20, k=3)        → 20×10×10    1,820 参数
    ↓ ReLU
Conv2D(20→20, k=3)        → 20×8×8      3,620 参数
    ↓ ReLU
MaxPool2D(2×2)            → 20×4×4
    ↓
Flatten                    → 320
    ↓ Dropout(0.2)
Linear(320→128)            → 128         41,088 参数
    ↓ ReLU
Linear(128→10)             → 10          1,290 参数
────────────────────────────────────────
总参数: ~49,000
```

## 训练结果

| Epoch | Train Loss | Train Acc | Test Acc | 时间 |
|-------|-----------|-----------|----------|------|
| 1 | 0.5448 | 82.6% | 96.8% | ~37s |
| 2 | 0.2366 | 92.9% | 97.7% | ~37s |
| 3 | 0.1874 | 94.4% | 97.9% | ~37s |
| 4 | 0.1678 | 94.9% | 98.3% | ~37s |
| 5 | 0.1466 | 95.6% | 98.4% | ~37s |

Batch size=64, lr=0.001, momentum=0.9, decay=0.95/epoch, 数据增强开启。~1700 samples/s at O3。

训练准确率低于测试准确率——数据增强让训练更难，模型被迫学习真正的形状不变性。

## 命令行用法

```bash
./build/mnist_cpp train      # 训练
./build/mnist_cpp predict    # 推理（stdin 读取 784 字节灰度值）
./build/mnist_cpp sanity     # 快速验证（128 样本）
./build/mnist_cpp export     # 导出权重为 JSON（输出到 stdout）
```

## 项目结构

```
src/
├── main.cpp                  # 入口（train / predict / sanity / export）
├── core/
│   ├── module.h              # Parameter, Module 基类
│   └── weight_io.h           # 权重保存/加载
├── data/mnist_loader.*       # MNIST IDX 格式解析
├── layers/
│   ├── conv2d.*              # 2D 卷积 (im2col + GEMM, batch)
│   ├── max_pool.*            # 2D 最大池化 (batch)
│   ├── flatten.*             # 展平层
│   ├── fully_connected.*     # 全连接层 (batch)
│   └── dropout.h             # Dropout 层 (header-only)
├── activations/relu.*        # ReLU 激活
├── loss/cross_entropy.*      # 交叉熵损失 (内置 softmax)
└── model/
    ├── model.*               # Sequential 容器
    └── trainer.*             # 训练循环 (mini-batch SGD)
docs/
├── index.html                # 静态手写识别页面（纯 JS 推理）
├── weights.json              # 模型权重
└── tutorial.md               # 架构设计原理
```

## 数据

需要 MNIST 原始 IDX 文件，放入 `data/` 目录：

- `train-images-idx3-ubyte`
- `train-labels-idx1-ubyte`
- `t10k-images-idx3-ubyte`
- `t10k-labels-idx1-ubyte`

可从 [Yann LeCun 的网站](https://yann.lecun.com/exdb/mnist/) 下载。

## 已知问题

- GCC 15.2.0 (MSYS2/MinGW) 的 `std::ifstream`/`std::ofstream` 在 O1+ 崩溃。`mnist_loader.cpp` O0 编译，权重 I/O 用 C FILE* 绕过
- Eigen 未链接 BLAS，矩阵乘速度为纯 C++ 实现
