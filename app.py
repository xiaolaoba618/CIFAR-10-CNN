"""
CIFAR-10 彩色图像分类 —— 拖拽识别小工具（桌面 App）

功能：
    把任意一张图片拖入窗口（或点击「选择图片」），程序会自动把图片缩放为 32×32，
    再用训练好的残差网络 residuals_cifar10_cnn.pth 推理，给出所属类别与置信度。

重要说明：
    模型训练时的输入是 32×32（见 utils.py 的 RandomCrop(32) 与 README）。
    因此这里必须同样缩放到 32×32 才能对齐训练分布。
    （实测：32×32 的测试准确率约 90%，而 64×64 仅约 22%。）

打包：
    见 README.md「打包为 exe」一节。
"""
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import filedialog, messagebox

from model import CIFAR_CNN

# 拖拽支持（tkinterdnd2 未安装时自动降级为「选择图片」按钮）
try:
    from tkinterdnd2 import DND_FILES, TkinterDnD
    HAS_DND = True
except Exception:                                     # pragma: no cover
    HAS_DND = False

INPUT_SIZE = 32                                       # 与训练一致，切勿改成 64（会严重掉点）
MODEL_FILE = "residuals_cifar10_cnn.pth"
MEAN = (0.4914, 0.4822, 0.4465)                       # CIFAR-10 官方 RGB 均值（与训练一致）
STD = (0.2023, 0.1994, 0.2010)                        # CIFAR-10 官方 RGB 标准差（与训练一致）

CLASSES_EN = ["airplane", "automobile", "bird", "cat", "deer",
              "dog", "frog", "horse", "ship", "truck"]
CLASSES_CN = ["飞机", "汽车", "鸟", "猫", "鹿", "狗", "青蛙", "马", "船", "卡车"]

# 设备选择（与 utils.py 保持一致，这里内联，使 app 不依赖 utils / torchvision）
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")


def preprocess(pil_img):
    """等价于 transforms.Compose([Resize((32,32)), ToTensor(), Normalize(MEAN, STD)])。

    这里用 PIL + torch、numpy 手写，刻意不依赖 torchvision，
    以免 PyInstaller 打包后出现 "operator torchvision::nms does not exist" 之类的问题。
    """
    img = pil_img.convert("RGB").resize((INPUT_SIZE, INPUT_SIZE), Image.BILINEAR)
    x = torch.from_numpy(np.asarray(img, dtype="float32") / 255.0)   # HWC, 取值 [0,1]
    x = x.permute(2, 0, 1).contiguous()                              # -> CHW
    mean = torch.tensor(MEAN, dtype=torch.float32).view(3, 1, 1)
    std = torch.tensor(STD, dtype=torch.float32).view(3, 1, 1)
    return (x - mean) / std


def resource_path(name):
    """兼容 PyInstaller 打包：优先从解压临时目录(_MEIPASS)找资源。"""
    base = getattr(sys, "_MEIPASS", os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base, name)


def load_model():
    """加载训练好的残差模型，返回 eval 模式下的模型。"""
    ckpt = torch.load(resource_path(MODEL_FILE), map_location=DEVICE)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        state_dict = ckpt["model_state_dict"]
        dropout = ckpt.get("dropout", 0.4)
    else:
        state_dict, dropout = ckpt, 0.4

    model = CIFAR_CNN(dropout=dropout).to(DEVICE)
    model.load_state_dict(state_dict)
    model.eval()
    return model


@torch.no_grad()
def predict(model, pil_img):
    """输入 PIL 图片，返回 (top1 索引, 概率列表)。"""
    x = preprocess(pil_img).unsqueeze(0).to(DEVICE)
    probs = F.softmax(model(x), dim=1)[0]
    top1 = int(probs.argmax().item())
    return top1, probs.cpu().tolist()


class App:
    def __init__(self, root):
        self.root = root
        self.model = load_model()

        root.title("CIFAR-10 图像识别 · 拖拽上传")
        root.geometry("560x680")
        root.configure(bg="#f4f5f7")
        root.resizable(False, False)

        # ---------- 标题 ----------
        tk.Label(root, text="CIFAR-10 彩色图像识别", font=("Microsoft YaHei", 16, "bold"),
                 bg="#f4f5f7", fg="#222222").pack(pady=(16, 2))
        tk.Label(root, text="把图片拖到下面方框里，自动识别它属于哪一类",
                 font=("Microsoft YaHei", 10), bg="#f4f5f7", fg="#666666").pack()

        # ---------- 拖拽区 ----------
        self.drop = tk.Label(
            root,
            text="📥\n\n把图片拖到这里\n\n（或点击下方「选择图片」）",
            font=("Microsoft YaHei", 12),
            bg="#ffffff", fg="#888888",
            width=38, height=6,
            relief="ridge", bd=2, cursor="hand2",
        )
        self.drop.pack(pady=14)
        if HAS_DND:
            self.drop.drop_target_register(DND_FILES)
            self.drop.dnd_bind("<<Drop>>", self.on_drop)
            self.drop.config(text="📥\n\n把图片拖到这里\n\n（或点击下方「选择图片」）")
        else:
            self.drop.config(text="（当前环境不支持拖拽，请用下方按钮选择图片）")

        # ---------- 预览 ----------
        self.preview = tk.Label(root, bg="#f4f5f7")
        self.preview.pack(pady=4)

        # ---------- 结果 ----------
        self.result = tk.Label(root, text="", font=("Microsoft YaHei", 13, "bold"),
                               bg="#f4f5f7", fg="#1a7f37", justify="center")
        self.result.pack(pady=(6, 2))

        self.detail = tk.Label(root, text="", font=("Consolas", 10),
                               bg="#f4f5f7", fg="#555555", justify="left")
        self.detail.pack()

        # ---------- 按钮 ----------
        tk.Button(root, text="选择图片", font=("Microsoft YaHei", 11),
                  width=14, height=2, bg="#2d7ff9", fg="white",
                  activebackground="#1f66cc", activeforeground="white",
                  relief="flat", cursor="hand2",
                  command=self.on_choose).pack(pady=14)

        self.path_label = tk.Label(root, text="", font=("Microsoft YaHei", 9),
                                   bg="#f4f5f7", fg="#999999", wraplength=500)
        self.path_label.pack()

    # ---------- 事件 ----------
    def on_drop(self, event):
        # event.data 可能是 "{C:/a b.jpg} C:/c.jpg" 形式，用 splitlist 正确解析
        try:
            paths = self.root.tk.splitlist(event.data)
        except Exception:
            paths = [event.data]
        if paths:
            self.handle(paths[0])

    def on_choose(self):
        path = filedialog.askopenfilename(
            title="选择一张图片",
            filetypes=[("图片文件", "*.jpg *.jpeg *.png *.bmp *.gif *.webp"),
                       ("所有文件", "*.*")],
        )
        if path:
            self.handle(path)

    def handle(self, path):
        try:
            img = Image.open(path)
            img.load()
        except Exception as e:
            messagebox.showerror("打不开这张图片", f"{path}\n\n{e}")
            return

        # 预览（等比缩放到 200×200 以内）
        thumb = img.convert("RGB").copy()
        thumb.thumbnail((200, 200), Image.LANCZOS)
        self._photo = ImageTk.PhotoImage(thumb)
        self.preview.config(image=self._photo)

        # 推理
        top1, probs = predict(self.model, img)
        name_cn, name_en = CLASSES_CN[top1], CLASSES_EN[top1]
        conf = probs[top1]

        self.result.config(text=f"识别结果：{name_cn}（{name_en}）  {conf*100:.1f}%")

        # 概率前 3 名
        order = sorted(range(10), key=lambda i: probs[i], reverse=True)[:3]
        lines = ["Top-3 置信度："]
        for i in order:
            bar = "█" * int(round(probs[i] * 30))
            lines.append(f"  {CLASSES_CN[i]:<3}{CLASSES_EN[i]:<11}{probs[i]*100:5.1f}%  {bar}")
        self.detail.config(text="\n".join(lines))

        self.path_label.config(text=os.path.basename(path))


def main():
    root = TkinterDnD.Tk() if HAS_DND else tk.Tk()
    try:
        App(root)
    except Exception as e:
        messagebox.showerror("启动失败", f"模型加载失败：{e}")
        raise
    root.mainloop()


if __name__ == "__main__":
    # PyInstaller 打包后多进程需 freeze_support（本 app 未用多进程，加上更稳妥）
    import multiprocessing
    multiprocessing.freeze_support()
    main()
