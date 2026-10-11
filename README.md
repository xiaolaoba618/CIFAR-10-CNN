# CIFAR-10 彩色图像分类（CNN · PyTorch）

在之前 MNIST 手写数字项目的基础上进阶：识别 **10 类彩色物体**（飞机、汽车、鸟、猫……）。
核心变化是 **从单通道灰度小图 → 三通道彩色图**，网络更深更宽，并引入了彩色图必需的**数据增强**。

## 项目结构

```
CNN进阶识别彩色图像/
├── model.py              # CNN 网络结构（CIFAR_CNN）
├── train.py              # 训练脚本（Warmup+余弦退火、EarlyStopping）
├── utils.py              # 数据加载、增强、设备选择、单轮训练/验证
├── test.py               # 加载权重，测试集评估
├── best_cifar10_cnn.pth  # 训练后生成的权重
└── data/                 # CIFAR-10 数据集
```

## 模型结构

三段「卷积 → BN → ReLU（×2）→ 最大池化」，末端用自适应池化接分类头：

```
输入 3×32×32
  [Conv(3→64) ×2 + BN + ReLU] + MaxPool     32→16
  [Conv(64→128) ×2 + BN + ReLU] + MaxPool   16→8
  [Conv(128→256) ×2 + BN + ReLU]
  AdaptiveAvgPool2d(1)   256×1×1
  Flatten → Linear(256→64) + ReLU + Dropout(0.4)
  Linear(64→10)   # 输出 logits
```

## 对比 MNIST 的改进

| 维度 | MNIST（之前） | CIFAR-10（本次） |
| --- | --- | --- |
| 任务 | 灰度手写数字，1×28×28 | 彩色物体，**3×32×32（RGB）** |
| 网络深度 | 2 段卷积，通道 32→64 | **3 段卷积，通道 64→128→256** |
| 分类头 | Flatten 64×7×7=3136 → 128 → 10 | **自适应平均池化 → 256 → 64 → 10（参数更少）** |
| Dropout | 0.25 | **0.4（彩色任务更易过拟合，加大正则）** |
| 数据增强 | 仅 ToTensor + Normalize | **RandomCrop / 水平翻转 / ColorJitter / RandomErasing** |
| 归一化 | 单通道 (0.1307,) | **三通道 CIFAR 官方 mean/std** |
| 学习率 | 自定义 WarmupCosineScheduler | **SequentialLR(LinearLR + CosineAnnealingLR) 组合** |
| 数据划分 | 先 transform 再 random_split | **transform=None 读原图，再套 `TransformSubset` 给训练/验证不同增强** |
| 数据读取 | 单进程 | **num_workers=4 + `if __name__ == '__main__'` 保护** |

**三个关键升级点：**

1. **输入从 1 通道变 3 通道** —— 卷积输入通道改为 3，归一化改用 CIFAR 官方 RGB 统计量。
2. **用增强对抗过拟合** —— 彩色图信息量大、更易过拟合，因此加入裁剪/翻转/色彩抖动/随机擦除；训练集用原图 + 随机增强，验证/测试集仅做标准化。
3. **末端自适应池化** —— 用 `AdaptiveAvgPool2d(1)` 替代大展平，显著压缩全连接参数量。

## 训练策略

- 优化器 AdamW，`lr=0.01`，`weight_decay=1e-2`
- 学习率：3 轮 Warmup + 余弦退火（`1e-5 → 0.01 → 1e-5`）
- 早停：验证损失连续 5 轮不下降即停止（上限 20 轮）
- 数据：45000 训练 / 5000 验证，按验证损失最低处保存权重

## 使用

> 环境复用 `CNN_MINIST/.venv`（torch 2.14.1+cu130 / torchvision 0.29.1+cu130）。

```bash
# 训练（生成 residuals_cifar10_cnn.pth）
python train.py

# 测试集评估
python test.py
```

## 环境

Python 3.13 · PyTorch · torchvision · numpy（GPU 可选，CPU 亦可运行）
