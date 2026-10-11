"""
CIFAR-10 测试 / 评估脚本。

加载 train.py 保存的最佳权重，在官方测试集（10000 张）上评估损失与准确率。

Windows 多进程安全：DataLoader 的构建与评估都放在 main() 里，
由 `if __name__ == '__main__':` 调用。
"""
import torch
import torch.nn as nn

from model import CIFAR_CNN
from utils import DEVICE, get_dataloaders, run_epoch

CKPT_PATH = "residuals_cifar10_cnn.pth"


def load_model(ckpt_path=CKPT_PATH, device=DEVICE):
    """加载训练好的模型，返回 eval 模式下的模型。"""
    ckpt = torch.load(ckpt_path, map_location=device)

    # 兼容两种保存格式
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        state_dict = ckpt["model_state_dict"]
        dropout = ckpt.get("dropout", 0.4)
    else:
        state_dict, dropout = ckpt, 0.4

    model = CIFAR_CNN(dropout=dropout).to(device)
    model.load_state_dict(state_dict)
    model.eval()          # 关键：关掉 Dropout / BatchNorm 的训练行为
    return model


def main():
    # 只需要 test_loader；其余两个 DataLoader 未被迭代，不会拉起子进程
    _, _, test_loader = get_dataloaders()

    model = load_model()
    criterion = nn.CrossEntropyLoss()

    test_loss, test_acc = run_epoch(model, test_loader, criterion, optimizer=None)
    print(f"Test loss={test_loss:.4f}, acc={test_acc:.4f}")


if __name__ == "__main__":
    main()
