# MNIST CNN in C++

纯 C++ (Eigen, 无深度学习框架) 实现的 MNIST 手写数字识别 CNN，**98.4% 测试准确率**。

详细架构设计原理见 [docs/tutorial.md](docs/tutorial.md)。

## 快速开始

```bash
# 1. 编译（需要 CMake + g++，自动下载 Eigen 3.4.0）
cd build && cmake -DCMAKE_BUILD_TYPE=Release .. && mingw32-make

# 2. 训练（生成 model_weights.bin，约 3 分钟）
./build/mnist_cpp.exe train

# 3. 启动 Web 验证
cd web && python server.py
# 浏览器打开 http://localhost:5000，在画布上写数字验证
```

需要 Python 依赖：`pip install flask`

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

CrossEntropyLoss 内部计算 softmax，模型最后一层不加 Softmax。

## 训练结果

| Epoch | Train Loss | Train Acc | Test Acc | 时间 |
|-------|-----------|-----------|----------|------|
| 1 | 0.5448 | 82.6% | 96.8% | ~37s |
| 2 | 0.2366 | 92.9% | 97.7% | ~37s |
| 3 | 0.1874 | 94.4% | 97.9% | ~37s |
| 4 | 0.1678 | 94.9% | 98.3% | ~37s |
| 5 | 0.1466 | 95.6% | 98.4% | ~37s |

Batch size=64, lr=0.001, momentum=0.9, decay=0.95/epoch, data augmentation active. ~1700 samples/s at O3.

注意训练准确率低于测试准确率，因为数据增强让训练更难，模型被迫学到真正的形状不变性。

## 训练配置

| 参数 | 值 |
|------|-----|
| 优化器 | SGD + Momentum (0.9) |
| 学习率 | 0.001，每轮衰减 0.95 |
| Batch size | 64 |
| Dropout | 0.2 (Flatten 后) |
| 数据增强 | 旋转 ±15°, 缩放 0.85-1.15, 平移 ±2.8px |

## 项目结构

```
src/
├── main.cpp                  # 入口（train / predict / sanity）
├── core/
│   ├── module.h              # Parameter, Module 基类
│   └── weight_io.h           # 权重保存/加载 (C FILE*)
├── data/mnist_loader.*       # MNIST IDX 格式解析
├── layers/
│   ├── conv2d.*              # 2D 卷积 (im2col + GEMM, batch)
│   ├── max_pool.*            # 2D 最大池化 (batch)
│   ├── flatten.*             # 展平层 (no-op in batch mode)
│   ├── fully_connected.*     # 全连接层 (batch)
│   └── dropout.h             # Dropout 层 (header-only)
├── activations/relu.*        # ReLU 激活
├── loss/cross_entropy.*      # 交叉熵损失 (内置 softmax)
└── model/
    ├── model.*               # Sequential 容器
    └── trainer.*             # 训练循环 (mini-batch SGD)
web/
├── server.py                 # Flask Web 服务
└── templates/index.html      # Canvas 画布前端
doc/
└── tutorial.md               # 架构设计原理
```

## 命令行用法

```bash
# 训练
./build/mnist_cpp.exe train

# 推理（从 stdin 读取 784 字节原始灰度值）
echo ... | ./build/mnist_cpp.exe predict

# 快速验证
./build/mnist_cpp.exe sanity
```

## 已知问题

- GCC 15.2.0 (MSYS2/MinGW) 的 `std::ifstream`/`std::ofstream` 在 O1+ 崩溃。`mnist_loader.cpp` O0 编译，权重 I/O 用 C FILE* 绕过。
- OpenMP 对此模型规模加速有限，保持关闭。
- Eigen 未链接 BLAS，矩阵乘速度为纯 C++ 实现。
