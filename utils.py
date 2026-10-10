"""
数据加载与训练辅助工具。

包含：随机种子 / 设备选择 / 数据增强 / DataLoader 构建 / 单个 epoch 的训练与验证。

Windows 多进程注意事项：
    本脚本的 DataLoader 统一封装在 get_dataloaders() 函数里，num_workers=4 会启动子进程；
    调用方（train.py / test.py）必须在 `if __name__ == '__main__':` 保护下调用本函数，
    否则子进程重新导入主模块时会再次构建数据集，导致循环执行（卡死或显存/内存暴涨）。
"""
import random
import numpy as np
import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

# %% 设置人工种子以便复现
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# %% 判断设备类型
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")

# %% 数据相关超参数
DATA_DIR = "./data"                 # 数据集根目录
BATCH_SIZE = 128                    # 单次参数更新使用 128 张图片
NUM_WORKERS = 4                     # 读取数据的子进程数（Windows 下需配合 __main__ 保护）
TRAIN_SIZE = 45000                  # 从 50000 张训练集里划出 45000 用作训练，其余作验证
MEAN = (0.4914, 0.4822, 0.4465)     # CIFAR-10 官方 RGB 三通道均值
STD = (0.2023, 0.1994, 0.2010)      # CIFAR-10 官方 RGB 三通道标准差

class TransformSubset(torch.utils.data.Dataset):
    """给 torch.utils.data.Subset 套一层 transform。

    原始 CIFAR10 以 transform=None 读取（返回 PIL 图像），在此处再施加增强 / 标准化，
    这样可以把同一份原始数据分别用于训练（增强）与验证 / 测试（仅标准化）的不同 transform。
    """

    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform

    def __getitem__(self, idx):
        # subset 返回的是 (PIL 图像, label)，在此处再做增强 / 标准化
        x, y = self.subset[idx]
        if self.transform is not None:
            x = self.transform(x)
        return x, y

    def __len__(self):
        return len(self.subset)


def get_dataloaders():
    """构建 train / val / test 三个 DataLoader 并返回 (train_loader, val_loader, test_loader)。

    Windows 注意：调用方必须在 `if __name__ == '__main__'` 保护下调用。
    """
    # 只在主进程打印一次环境信息（num_workers 的子进程会重复 import 本模块）
    print("PyTorch:", torch.__version__)
    print("Device:", DEVICE)
    print(f"num_workers: {NUM_WORKERS}")

    # 1. 加载原始数据（不做任何 transform）
    train_dataset = datasets.CIFAR10(root=DATA_DIR, train=True, download=True, transform=None)
    test_dataset = datasets.CIFAR10(root=DATA_DIR, train=False, download=True, transform=None)

    # 2. 从训练集划出训练 / 验证两部分
    val_size = len(train_dataset) - TRAIN_SIZE
    train_subset, val_subset = random_split(
        train_dataset, [TRAIN_SIZE, val_size],
        generator=torch.Generator().manual_seed(SEED),
    )

    # 3. 定义 transform：训练用随机增强，验证 / 测试只做标准化
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(0.2, 0.2, 0.2, 0.05),
        transforms.ToTensor(),
        transforms.RandomErasing(p=0.25),
        transforms.Normalize(MEAN, STD),
    ])
    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])

    # 4. 包装
    train_dataset = TransformSubset(train_subset, transform=train_transform)
    val_dataset = TransformSubset(val_subset, transform=val_transform)
    test_dataset = TransformSubset(test_dataset, transform=val_transform)

    # 5. DataLoader（num_workers=4，仅在迭代时才会真正拉起子进程）
    loader_kwargs = dict(batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, pin_memory=True,persistent_workers=True)
    train_loader = DataLoader(train_dataset, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_dataset, shuffle=False, **loader_kwargs)

    return train_loader, val_loader, test_loader


def run_epoch(model, loader, criterion, optimizer=None, device=DEVICE):
    """跑一个 epoch：optimizer 不为 None 时训练，否则只做验证 / 评估。

    返回 (平均损失, 平均准确率)。
    """
    training = optimizer is not None
    model.train() if training else model.eval()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)

        if training:
            optimizer.zero_grad()

        with torch.set_grad_enabled(training):
            logits = model(x)
            loss = criterion(logits, y)
            if training:
                loss.backward()
                optimizer.step()

        n = y.size(0)
        total_loss += loss.item() * n
        total_correct += (logits.argmax(dim=1) == y).sum().item()
        total_samples += n

    return total_loss / total_samples, total_correct / total_samples
