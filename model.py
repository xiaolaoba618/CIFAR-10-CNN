"""
定义 CIFAR-10 彩色图像分类的卷积神经网络主模型。

结构：3 段「卷积-卷积-池化」特征提取 + 自适应平均池化 + 全连接分类头。
输入 32x32 彩色图，输出 10 类 logits（不做 softmax）。
"""
import torch
import torch.nn as nn


class CIFAR_CNN(nn.Module):
    def __init__(self, dropout=0.4):
        super().__init__()

        # 卷积部分
        self.features = nn.Sequential(
            # 32 * 32
            nn.Conv2d(3, 128, kernel_size=3, padding=1),   # 输入 3 通道彩色图，输出 64 张特征图，尺寸不变
            nn.BatchNorm2d(128),                           # BatchNorm 标准化
            nn.ReLU(),                                    # 增加非线性
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),                  # 最大池化，32 -> 16
            # 16 * 16
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),                  # 16 -> 8
            # 8 * 8
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)                       # 把 512*8*8 压成 512*1*1
        )

        # 全连接部分
        self.classifier = nn.Sequential(
            nn.Flatten(),                                 # 展平最后特征图
            nn.Linear(512, 128),                           # 全连接，512 维映射到 128 维
            nn.ReLU(),
            nn.Dropout(p=dropout),                        # 随机 dropout 神经元，抑制过拟合
            nn.Linear(128, 10),                            # 输出层，映射为 10 类分类
        )

        self._initial_weights()

    def _initial_weights(self):
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):  # 只对含参数的卷积层与线性层初始化
                nn.init.kaiming_normal_(module.weight, nonlinearity="relu")  # kaiming 初始化
                if module.bias is not None:
                    nn.init.zeros_(module.bias)          # 偏置初始化为 0

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)                         # 返回 logits，而不是 softmax
