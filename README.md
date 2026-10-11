# CIFAR-10 彩色图像分类（ResNet 风格 · PyTorch）

在之前 MNIST 手写数字项目的基础上进阶：识别 **10 类彩色物体**（飞机、汽车、鸟、猫……）。  
核心变化是 **从单通道灰度小图 → 三通道彩色图**，网络更深更宽，并引入 **残差连接** 与彩色图必需的**数据增强**。

## 项目结构

```
CNN进阶识别彩色图像/
├── model.py              # 残差网络结构（CIFAR_CNN，含 BasicBlock）
├── train.py              # 训练脚本（Warmup+余弦退火、EarlyStopping）
├── utils.py              # 数据加载、增强、设备选择、单轮训练/验证
├── test.py               # 加载权重，测试集评估
├── residuals_cifar10_cnn.pth  # 训练后生成的残差模型权重
└── data/                 # CIFAR-10 数据集
```

## 模型结构

采用 **ResNet 风格** 的残差网络：3 个 stage，每个 stage 由多个 `BasicBlock` 堆叠，末端用自适应池化接分类头。

```
输入 3×32×32
  [Conv(3→32, 3×3) + BN + ReLU]                       32×32
  Layer1: 3 × BasicBlock(32→64, 首个 stride=2)         16×16
  Layer2: 3 × BasicBlock(64→128, 首个 stride=2)         8×8
  Layer3: 3 × BasicBlock(128→256, 首个 stride=1)        8×8
  AdaptiveAvgPool2d(1)   256×1×1
  Flatten → Linear(256→64) + ReLU + Dropout(0.4)
  Linear(64→10)   # 输出 logits
```

`BasicBlock` 内部结构：

```
输入 x
  out = ReLU(BN(Conv3x3(x)))
  out = BN(Conv3x3(out))
  shortcut = x（通道/尺寸不变） 或 Conv1x1+BN(x)（通道/尺寸变化）
  out = out + shortcut
  out = ReLU(out)
```

- 每个 `BasicBlock` 包含两个 3×3 卷积，`bias=False`，后接 BatchNorm。
- 当输入输出通道或空间尺寸不一致时，shortcut 用 1×1 卷积 + BN 进行匹配。
- 默认 `num_blocks=(3,3,3)`，即 3 个 stage 各 3 个残差块，共 9 个 BasicBlock。
- 分类头：`Linear(256→64) + ReLU + Dropout(0.4) + Linear(64→10)`。

## 对比普通 CNN 的改进

| 维度 | 普通 CNN（之前） | 残差网络（本次） |
| --- | --- | --- |
| 网络深度 | 3 段「卷积×2 + 池化」 | **3 个 stage，每个 stage 多个 BasicBlock，共 9 个残差块** |
| 连接方式 | 直筒式堆叠 | **引入残差连接，缓解梯度消失，易于更深网络训练** |
| 下采样 | 每段后接最大池化 | **通过 stride=2 的卷积进行下采样，shortcut 同步匹配** |
| 分类头 | Flatten 大展平 → 全连接 | **自适应平均池化 → 256 → 64 → 10（参数更少）** |
| Dropout | 0.25 | **0.4（彩色任务更易过拟合，加大正则）** |
| 数据增强 | 仅 ToTensor + Normalize | **RandomCrop / 水平翻转 / ColorJitter / RandomErasing** |
| 归一化 | 单通道 (0.1307,) | **三通道 CIFAR 官方 mean/std** |
| 学习率 | 自定义 WarmupCosineScheduler | **SequentialLR(LinearLR + CosineAnnealingLR) 组合** |
| 数据划分 | 先 transform 再 random_split | **transform=None 读原图，再套 `TransformSubset` 给训练/验证不同增强** |
| 数据读取 | 单进程 | **num_workers=4 + `if __name__ == '__main__'` 保护** |

**三个关键升级点：**

1. **输入从 1 通道变 3 通道** —— 卷积输入通道改为 3，归一化改用 CIFAR 官方 RGB 统计量。
2. **引入残差连接** —— 每个 `BasicBlock` 学习残差映射 `F(x)+x`，使网络更容易优化，支持更深结构。
3. **用增强对抗过拟合** —— 彩色图信息量大、更易过拟合，因此加入裁剪/翻转/色彩抖动/随机擦除；训练集用原图 + 随机增强，验证/测试集仅做标准化。

## 训练策略

- 优化器：AdamW，`lr=1e-2`，`weight_decay=1e-3`
- 学习率：5 轮 Warmup（从 `1e-6` 线性升至 `1e-2`） + 余弦退火至 `1e-5`，总日程 45 轮
- 早停：验证损失连续 10 轮不下降即停止（上限 45 轮）
- 数据：45000 训练 / 5000 验证，按验证损失最低处保存权重
- 保存文件：`residuals_cifar10_cnn.pth`

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