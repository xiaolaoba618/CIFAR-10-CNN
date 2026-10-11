"""
CIFAR-10 训练脚本。

思路搬运自 notebook：
    - Warm-up + 余弦退火学习率
    - 以验证损失为准的 Early-Stopping
    - 保存验证损失最低的模型权重

Windows 多进程安全：
    所有涉及 DataLoader（num_workers）的逻辑都放在 main() 里，
    并通过 `if __name__ == '__main__':` 调用，避免子进程重新执行训练流程造成循环执行。
"""
import copy
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

from model import CIFAR_CNN
from utils import DEVICE, get_dataloaders, run_epoch

# %% 训练超参数
DROPOUT = 0.4
WARMUP_EPOCHS = 5
MAX_LR = 1e-2
MIN_LR = 1e-5
WEIGHT_DECAY = 1e-3
TOTAL_EPOCHS = 45         # 余弦退火的总日程
MAX_EPOCHS = 45            # 实际最多训练轮数
PATIENCE = 10            # 验证损失连续 PATIENCE 轮不下降就早停
CKPT_PATH = "residuals_cifar10_cnn.pth"


def build_scheduler(optimizer):
    """Warm-up（线性升温） + 余弦退火 组合学习率。"""
    warmup_scheduler = LinearLR(
        optimizer,
        start_factor=1e-4,
        end_factor=1.0,
        total_iters=WARMUP_EPOCHS,
    )
    cosine_scheduler = CosineAnnealingLR(
        optimizer,
        T_max=TOTAL_EPOCHS - WARMUP_EPOCHS,
        eta_min=MIN_LR,
    )
    scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[WARMUP_EPOCHS],
    )
    return scheduler


def main():
    # 构建数据（必须在 __main__ 保护下调用，num_workers=4 会在迭代时拉起子进程）
    train_loader, val_loader, _ = get_dataloaders()

    model = CIFAR_CNN(dropout=DROPOUT).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=MAX_LR, weight_decay=WEIGHT_DECAY)
    scheduler = build_scheduler(optimizer)

    history = {
        "train_loss": [], "val_loss": [],
        "train_acc": [], "val_acc": [], "lr": []
    }
    best_val_loss = float("inf")
    best_state = None
    epochs_without_improvement = 0

    print("训练开始")
    for epoch in range(MAX_EPOCHS):
        lr_now = optimizer.param_groups[0]["lr"]

        train_loss, train_acc = run_epoch(
            model, train_loader, criterion, optimizer=optimizer
        )
        val_loss, val_acc = run_epoch(
            model, val_loader, criterion, optimizer=None
        )

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)
        history["lr"].append(lr_now)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        print(
            f"Epoch {epoch+1:02d}/{MAX_EPOCHS} | lr={lr_now:.2e} | "
            f"train loss={train_loss:.4f}, acc={train_acc:.4f} | "
            f"val loss={val_loss:.4f}, acc={val_acc:.4f}"
        )

        scheduler.step()

        if epochs_without_improvement >= PATIENCE:
            print("Early stopping: 验证损失连续多轮未改善。")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
        print(f"已恢复验证损失最低的模型，best val loss={best_val_loss:.4f}")

        checkpoint = {
            "model_state_dict": best_state,
            "best_val_loss": best_val_loss,
            "dropout": DROPOUT,        # 记录结构超参，加载时保持一致
            "epoch": epoch + 1,
            "max_epochs": MAX_EPOCHS,
        }
        torch.save(checkpoint, CKPT_PATH)
        print(f"已保存到 {CKPT_PATH}")


if __name__ == "__main__":
    main()
