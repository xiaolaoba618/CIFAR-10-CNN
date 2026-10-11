"""
定义 CIFAR-10 彩色图像分类的卷积神经网络主模型。

结构：3 段「卷积-卷积-池化」特征提取 + 自适应平均池化 + 全连接分类头。
输入 32x32 彩色图，输出 10 类 logits（不做 softmax）。
"""
import torch
import torch.nn as nn


class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self,in_channels,out_channels,stride=1):
        super().__init__()

        #F(x)卷积层
        self.Conv1 = nn.Conv2d(in_channels,out_channels,kernel_size=3,stride=stride,padding=1,bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.Conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1,bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        #x传播层
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.shortcut = nn.Sequential()

    def forward(self, x):
        out = self.relu(self.bn1(self.Conv1(x)))        #等价于一个残差块内第一个卷积操作
        out = self.bn2(self.Conv2(out))                 #第二个卷积操作
        out = out + self.shortcut(x)                    #F（X）+ X
        return self.relu(out)

class CIFAR_CNN(nn.Module):
    def __init__(self, block = BasicBlock,num_blocks = (3,3,3),dropout=0.4):
        super().__init__()
        self.in_channels = 32

        self.stem = nn.Sequential(
            nn.Conv2d(3, self.in_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(self.in_channels),
            nn.ReLU(inplace=True)
        )

        # 残差层
        self.Layer1 = self._make_layer(block, 64,num_blocks[0], stride=2)
        self.Layer2 = self._make_layer(block,128,num_blocks[1], stride=2)
        self.Layer3 = self._make_layer(block,256,num_blocks[2], stride=1)
        #池化
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))



        # 全连接部分
        self.classifier = nn.Sequential(                              # 展平最后特征图
            nn.Linear(256 * block.expansion, 64),                           # 全连接，512 维映射到 128 维
            nn.ReLU(),
            nn.Dropout(p=dropout),                        # 随机 dropout 神经元，抑制过拟合
            nn.Linear(64, 10),                            # 输出层，映射为 10 类分类
        )

        self._initial_weights()

    def _initial_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)          # 偏置初始化为 0

    def _make_layer(self, block, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)  # 获得一个列表，保证了每一个layer第一个残差块stride进行降分辨率，例如[2,1,1,1]
        layers = []
        for s in strides:
            layers.append(block(self.in_channels, out_channels, s))
            self.in_channels = out_channels * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.stem(x)                                          #第一次主干卷积
        out = self.Layer1(out)                                      #引入残差层
        out = self.Layer2(out)
        out = self.Layer3(out)
        out = self.avgpool(out)                                     #拍平pool，每个通道对应一个数
        out = torch.flatten(out, 1)
        out = self.classifier(out)
        return out                         # 返回 logits，而不是 softmax
