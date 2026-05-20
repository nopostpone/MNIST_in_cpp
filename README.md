# MNIST CNN 层原理

## 1. Flatten 层（展平）

最简单的层，负责将多维张量"拍平"成一维向量，为后续全连接层做准备。

```
输入:  (C, H, W)  通道 × 高 × 宽   例: (64, 7, 7)
输出:  (C*H*W, )  一维向量         例: (3136, )

操作: 保持数值不变，仅改变形状。无参数，无需反向传播。
```

在卷积层和全连接层之间充当"翻译官"。

---

## 2. 全连接层（Fully Connected / Linear）

### 前向传播

```
y = W·x + b

输入 x:  (N_in,  )    → 上一层的输出（或 Flatten 后的向量）
权重 W:  (N_out, N_in) → 可训练参数
偏置 b:  (N_out, )    → 可训练参数
输出 y:  (N_out, )    → 对下层的输入
```

每个输出神经元与所有输入神经元相连，权重矩阵 W 的每一行对应一个输出神经元。

### 反向传播

```
loss 对输出的梯度:  dy        (已知，从下一层传来)
loss 对参数的梯度:  dW = dy · x^T       (N_out × N_in)
                   db = dy              (N_out × 1)

loss 对输入的梯度:  dx = W^T · dy       (传回上一层)
```

### 参数更新 (SGD)

```
W := W - lr * dW
b := b - lr * db
```

---

## 3. 二维卷积层（Conv2D）

### 前向传播

用一个可学习的卷积核（kernel）在输入特征图上滑动，计算点积。

```
输入:  (1, H_in, W_in)  单通道，28×28 的 MNIST 图像
输出:  (C_out, H_out, W_out)

H_out = (H_in - K + 2*P) / S + 1
W_out = (W_in - K + 2*P) / S + 1

K:  卷积核大小 (kernel_size)，如 3 或 5
P:  填充       (padding)，通常 SAME 或 VALID
S:  步长       (stride)
```

### 具体计算（以第一层为例）

输入是 `28×28` 的单通道图像，使用 `3×3` 卷积核，`padding=0, stride=1`：

```
输出尺寸: (28 - 3 + 0) / 1 + 1 = 26×26

每个输出位置 (i, j) 的值:
  output[k][i][j] = sum_{m=0..K-1} sum_{n=0..K-1} input[i+m][j+n] * kernel[k][m][n]

加上偏置: output[k][i][j] += bias[k]
```

对于 MNIST：
- **Conv1**: `1×28×28` → 32 个 `3×3` 核 → `32×26×26`，参数量 = 32×3×3 + 32 = **320**
- **Conv2**: `32×13×13` → 64 个 `3×3` 核 → `64×11×11`，参数量 = 64×32×3×3 + 64 = **18,496**

### 反向传播

卷积核梯度：输入与输出梯度的互相关。

```
对 kernel 的梯度:
  dW[k][m][n] = sum_{i,j} input[i+m][j+n] * d_out[k][i][j]

对 input 的梯度 (传回上一层):
  将 d_out 与翻转 180° 的 kernel 进行完整卷积 (full convolution)
  → 即 padding=K-1 的转置卷积
```

---

## 4. 最大池化层（MaxPool2D）

### 前向传播

在局部窗口内取最大值，降低特征图尺寸，减少计算量并提供平移不变性。

```
输入:  (C, H_in, W_in)     例: (32, 26, 26)
输出:  (C, H_out, W_out)   例: (32, 13, 13)

H_out = H_in / pool_size    (通常 pool_size=2, stride=2)
W_out = W_in / pool_size

每个 2×2 窗口:
  output[k][i][j] = max( input[k][2i][2j],       input[k][2i][2j+1],
                         input[k][2i+1][2j],     input[k][2i+1][2j+1] )
```

无参数。

### 反向传播

梯度只传递给前向时取得最大值的位置（max pooling 的"路由"效应）。

```
对于每个输出位置，前向时我们记录了最大值的位置 (rmax, cmax):
  d_input[k][rmax][cmax] += d_output[k][i][j]
  其他位置梯度 = 0
```

---

## 5. 激活函数

### ReLU（卷积层后使用）

```
前向: y = max(0, x)
反向: dy/dx = 1 if x > 0 else 0
```

简单高效，缓解梯度消失。无参数。

### Softmax（最后一层使用）

将输出 logits 转换为概率分布。

```
前向: y_i = e^{x_i} / sum_j e^{x_j}      使得 sum_i y_i = 1
反向: dy_i/dx_j = y_i * (δ_ij - y_j)     δ_ij 是克罗内克函数
```

结合交叉熵损失时梯度简化为 `y_true - y_pred`。

---

## 6. 交叉熵损失（Cross Entropy Loss）

```
L = -sum_c y_true[c] * log(y_pred[c])

其中 y_true 是 one-hot 标签，y_pred 是 softmax 输出。

对于正确类别 k:  L = -log(y_pred[k])

梯度: dL/dx = y_pred - y_true    (结合 softmax + cross entropy 的简洁形式)
```

---

## 整体架构回顾

```
Input(1×28×28)
    ↓
Conv2D(1→32, k=3) ─── 320 参数
    ↓
ReLU
    ↓
MaxPool2D(2×2)      ─── 0  参数, 输出 32×13×13
    ↓
Conv2D(32→64, k=3)  ─── 18,496 参数
    ↓
ReLU
    ↓
MaxPool2D(2×2)      ─── 0  参数, 输出 64×7×7
    ↓
Flatten             ─── 0  参数, 输出 3136
    ↓
Linear(3136→128)    ─── 401,408 参数
    ↓
ReLU
    ↓
Linear(128→10)      ─── 1,290 参数
    ↓
Softmax
    ↓
Output(10)
────────────────────────────
总参数: ~421,000
```
