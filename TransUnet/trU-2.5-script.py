#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""TransUNet 2.5D training script (from notebook trU-2.5.ipynb, reorganized for script use).

- 保留原 notebook 的所有功能：数据准备、2.5D Dataset、TransUNet 构建、训练循环、
  2D/3D 评价以及可视化。
- 主要改动：把训练等主流程放到 main() 和 if __name__ == '__main__' 保护下，
  方便在 Windows + 多进程 DataLoader 下运行。
"""

# ================== imports & config ==================
import os
import random
from glob import glob

import numpy as np
import pandas as pd
from PIL import Image
import cv2
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler

import albumentations as A
import torch.nn.functional as F

import sys
sys.path.append(".")

from networks.vit_seg_modeling import VisionTransformer as ViT_seg
from networks.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg

import networks.vit_seg_modeling as vit_seg_modeling
import networks.vit_seg_modeling_resnet_skip as vit_seg_resnet_skip

# ======== ViT backbone 列表（方便大作业多模型对比） ========
VIT_BACKBONES = {
    # 1) ResNet50 + ViT-B_16
    "R50-ViT-B_16": {
        "patch_size": 16,
        "pretrained_path": r"F:\bmeml\dzy\pretrain\imagenet21k\imagenet21k_R50+ViT-B_16.npz",
    },
    # 2) 纯 ViT-B_16
    "ViT-B_16": {
        "patch_size": 16,
        "pretrained_path": r"F:\bmeml\dzy\pretrain\imagenet21k\imagenet21k_ViT-B_16.npz",
    },
    # 3) 纯 ViT-L_16
    "ViT-L_16": {
        "patch_size": 16,
        "pretrained_path": r"F:\bmeml\dzy\pretrain\imagenet21k\imagenet21k_ViT-L_16.npz",
    },
    # 4) ResNet50 + ViT-L_32
    "R50-ViT-L_32": {
        "patch_size": 32,
        "pretrained_path": r"F:\bmeml\dzy\pretrain\imagenet21k\imagenet21k_R50+ViT-L_32.npz",
    },
    # 5) ResNet26 + ViT-B_32
    "R26-ViT-B_32": {
        "patch_size": 32,
        "pretrained_path": r"F:\bmeml\dzy\pretrain\imagenet21k\imagenet21k_R26+ViT-B_32.npz",
    },
    # 5) ResNet26 + ViT-B_32
    "ViT-B_32": {
        "patch_size": 32,
        "pretrained_path": r"F:\bmeml\dzy\pretrain\imagenet21k\imagenet21k_ViT-B_32.npz",
    },
}

class CFG:
    # ======== 路径 ========
    data_root = r"F:\bmeml\dzy\uw-madison-gi-tract-image-segmentation"
    train_csv = os.path.join(data_root, "train.csv")
    train_dir = os.path.join(data_root, "train")
    out_dir   = "./checkpoints_transunet_2_5D"

    # ======== 图像 & 标签 ========
    img_size    = 256
    in_channels = 3          # 2.5D：[-1,0,+1] 3 通道
    num_classes = 3          # large_bowel, small_bowel, stomach

    # ======== 2.5D 设置 ========
    slice_window = [-1, 0, 1]

    # ======== 训练 ========
    seed         = 42
    batch_size   = 4
    num_workers  = 4
    lr           = 1e-4
    weight_decay = 1e-5
    num_epochs   = 15
    device       = "cuda" if torch.cuda.is_available() else "cpu"
    use_amp = True

    # ======== 学习率调度 ========
    use_scheduler = True      # 开 / 关 scheduler
    lr_step_size  = 5         # 每多少个 epoch 衰减一次
    lr_gamma      = 0.5       # 每次衰减的倍数（0.5=减半）

    # ======== 数据划分（新增） ========
    train_case_ratio = 0.8  # case_day 级别 train/val 比例（默认 8:2）
    neg_keep_ratio = 0.2  # 训练集中保留多少“全空 case_day”，0.3=只留30%

    # ======== TransUNet / ViT ========
    vit_name         = "ViT-B_16"
    vit_patches_size = 16
    n_skip           = 0
    use_pretrained   = True# 没有 npz 的话可以先改成 False

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


set_seed(CFG.seed)
os.makedirs(CFG.out_dir, exist_ok=True)

print("Using device:", CFG.device)
print("Data root:", CFG.data_root)

# 为了让后面的函数能访问，先声明若干“全局占位符”
model = None
train_loader = None
val_loader = None
val_dataset = None
dice_loss_fn = None
bce_loss_fn = None
optimizer = None
best_ckpt_path = None

# ================== 修复 TransUNet 在 Windows 下加载 npz 的问题 ==================
def npz_pjoin(*parts):
    """npz 里的 key 一律用 '/' 分隔，不能用 Windows 的 '\\'。"""
    return "/".join(parts)


# 覆盖两个模块里的 pjoin 引用
vit_seg_modeling.pjoin = npz_pjoin
vit_seg_resnet_skip.pjoin = npz_pjoin

# ================== RLE, Dice, enhance ==================

# 按 large_bowel / small_bowel / stomach 定义可视化颜色 (R,G,B 0~1)
COLORS = [
    [0.9, 0.0, 0.0],  # large_bowel
    [0.0, 0.8, 0.0],  # small_bowel
    [0.0, 0.0, 0.9],  # stomach
]


def rle_decode(mask_rle, shape):
    """从 train.csv 的 RLE 字符串解码为 2D mask，shape: (H, W)。"""
    if str(mask_rle) == 'nan' or mask_rle == '':
        return np.zeros(shape, dtype=np.uint8)

    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0::2], s[1::2])]
    starts -= 1
    ends = starts + lengths

    # Kaggle 医学分割的 RLE 是 Column-major (Fortran-style)
    img = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1

    # 关键
    return img.reshape(shape, order='C')



def compute_dice_2d_all(pred, gt):
    """
    原始定义：所有 slice 都参与：
      - pred, gt 全 0 => Dice = 1.0
      - 一边全 0 => Dice = 0.0
    """
    pred = pred.astype(np.uint8)
    gt   = gt.astype(np.uint8)

    pred_sum = np.sum(pred)
    gt_sum   = np.sum(gt)

    if pred_sum == 0 and gt_sum == 0:
        return 1.0
    if pred_sum == 0 or gt_sum == 0:
        return 0.0

    inter = np.sum(pred & gt)
    return (2.0 * inter) / (pred_sum + gt_sum + 1e-8)


def compute_dice_2d_pos(pred, gt):
    """
    positive 定义：只在 GT 有器官时参与平均；
      - GT 全 0 且 pred 全 0 => 返回 None（忽略这一对）
      - GT 全 0 且 pred 非 0 => 误检，Dice = 0
      - GT 非 0 且 pred 全 0 => 漏检，Dice = 0
    """
    pred = pred.astype(np.uint8)
    gt   = gt.astype(np.uint8)

    pred_sum = np.sum(pred)
    gt_sum   = np.sum(gt)

    if gt_sum == 0:
        if pred_sum == 0:
            return None      # 全背景，忽略
        else:
            return 0.0       # 有误检，记 0

    if pred_sum == 0:
        return 0.0           # GT 有器官但预测空，记 0

    inter = np.sum(pred & gt)
    return (2.0 * inter) / (pred_sum + gt_sum + 1e-8)

def enhance_image_natural(img_array):
    """更自然的 MRI 显示：截断高亮，线性拉伸到 0~1。"""
    img = img_array.astype(np.float32)
    p_high = np.percentile(img, 99.5)
    img = np.clip(img, 0, p_high)
    img = (img - img.min()) / (img.max() - img.min() + 1e-8)
    return img

# ================== build path maps & CSV ==================

train_images = glob(os.path.join(CFG.train_dir, "**", "*.png"), recursive=True)

id2path = {}          # "case123_day20_slice_0001" -> png path
case_day2slices = {}  # "case123_day20" -> [1,2,...]

for path in train_images:
    norm_path = path.replace("\\", "/")
    parts = norm_path.split("/")
    filename = parts[-1]

    stem = filename.split(".")[0]
    tokens = stem.split("_")
    if len(tokens) < 2 or tokens[0] != "slice":
        raise ValueError(f"Unexpected filename pattern: {filename}")
    slice_num = int(tokens[1])

    case_like = [p for p in parts if p.startswith("case")]
    if len(case_like) < 2:
        raise ValueError(f"Unexpected path (cannot find 'caseXXX_dayYY'): {norm_path}")
    case_day = case_like[1]

    img_id = f"{case_day}_slice_{slice_num:04d}"

    id2path[img_id] = norm_path
    case_day2slices.setdefault(case_day, []).append(slice_num)

for k in case_day2slices:
    case_day2slices[k] = sorted(case_day2slices[k])

print("Total png images:", len(id2path))
print("Total case_day:", len(case_day2slices))


def parse_case_day_slice(img_id: str):
    """"case123_day20_slice_0033" -> ("case123_day20", 33)。"""
    parts = img_id.split("_")
    case_day = "_".join(parts[:2])
    slice_num = int(parts[-1])
    return case_day, slice_num


# 读取 train.csv
df_all = pd.read_csv(CFG.train_csv)
df_all = df_all.fillna("")

print("CSV rows:", len(df_all))
print("Unique ids:", df_all["id"].nunique())

CLASS2IDX = {
    "large_bowel": 0,
    "small_bowel": 1,
    "stomach": 2,
}
IDX2CLASS = {v: k for k, v in CLASS2IDX.items()}


# ================== 2.5D Dataset ==================
class UWGITract2_5D_Dataset(Dataset):
    """2.5D UW-Madison GI Dataset。"""

    def __init__(self, df: pd.DataFrame, transforms=None):
        super().__init__()
        self.df = df
        self.transforms = transforms
        self.ids = sorted(df["id"].unique())

    def __len__(self):
        return len(self.ids)

    def _get_neighbor_id(self, case_day, slice_num, offset):
        all_slices = case_day2slices[case_day]
        target = slice_num + offset
        if target < all_slices[0]:
            target = all_slices[0]
        if target > all_slices[-1]:
            target = all_slices[-1]
        return f"{case_day}_slice_{target:04d}"

    def __getitem__(self, idx):
        img_id = self.ids[idx]
        case_day, slice_num = parse_case_day_slice(img_id)

        # 2.5D 图像：前一张 + 当前 + 后一
        imgs = []
        for off in CFG.slice_window:
            nid = self._get_neighbor_id(case_day, slice_num, off)
            path = id2path.get(nid, None)
            if path is None:
                raise RuntimeError(f"No png found for {nid}")
            img_raw = cv2.imread(path, cv2.IMREAD_UNCHANGED)
            if img_raw is None:
                raise RuntimeError(f"Failed to read {path}")
            if img_raw.ndim == 3:
                img_raw = img_raw[..., 0]
            imgs.append(img_raw.astype(np.float32))

        img_stack = np.stack(imgs, axis=0)  # [3, H, W]
        H, W = img_stack.shape[1], img_stack.shape[2]

        # 构造 3 通道 GT mask
        mask_3c = np.zeros((CFG.num_classes, H, W), dtype=np.uint8)
        rows = self.df[self.df["id"] == img_id]

        for _, row in rows.iterrows():
            cls_name = row["class"]
            rle = row["segmentation"]
            if cls_name in CLASS2IDX and rle != "":
                c = CLASS2IDX[cls_name]
                mask_3c[c] = rle_decode(rle, (H, W))

        # resize 到 CFG.img_size
        if (H != CFG.img_size) or (W != CFG.img_size):
            img_resized = []
            for c in range(img_stack.shape[0]):
                im = Image.fromarray(img_stack[c])
                im = im.resize((CFG.img_size, CFG.img_size), Image.BILINEAR)
                img_resized.append(np.array(im, dtype=np.float32))
            img_stack = np.stack(img_resized, axis=0)

            mask_resized = np.zeros((CFG.num_classes, CFG.img_size, CFG.img_size), dtype=np.uint8)
            for c in range(CFG.num_classes):
                m = Image.fromarray(mask_3c[c])
                m = m.resize((CFG.img_size, CFG.img_size), Image.NEAREST)
                mask_resized[c] = np.array(m, dtype=np.uint8)
            mask_3c = mask_resized

        # 简单归一化
        min_v, max_v = img_stack.min(), img_stack.max()
        if max_v > min_v:
            img_stack = (img_stack - min_v) / (max_v - min_v + 1e-8)
        else:
            img_stack = img_stack / 65535.0

        # Albumentations 增强（可选）
        if self.transforms is not None:
            img_hwc = np.transpose(img_stack, (1, 2, 0))
            mask_hwc = np.transpose(mask_3c, (1, 2, 0))
            aug = self.transforms(image=img_hwc, mask=mask_hwc)
            img_stack = np.transpose(aug["image"], (2, 0, 1))
            mask_3c = np.transpose(aug["mask"], (2, 0, 1))

        img_tensor = torch.from_numpy(img_stack).float()
        mask_tensor = torch.from_numpy(mask_3c).long()

        return img_tensor, mask_tensor, img_id

# ================== 创建 Dataset & DataLoader ==================
def create_datasets_and_loaders():
    """
    划分逻辑：
      1) 先按 case_day 做一次整体的 train/val 划分（不看 mask）；
      2) 只在“已经划入训练集”的样本中，以切片(id)为单位区分：
           - pos_ids_train: 至少有一个非空 mask 的 slice
           - neg_ids_train: 所有类都没有 mask 的 slice
         然后按 CFG.neg_keep_ratio 随机保留一部分空 slice；
      3) 验证集保持原始分布（不删空 slice）。
    """
    df = df_all.copy()
    # 每一行是否有 mask（注意是一行 = 一个类，不是整张图）
    df["has_mask_row"] = df["segmentation"].astype(str) != ""
    # case_day: "case123_day20_slice_0033" -> "case123_day20"
    df["case_day"] = df["id"].apply(lambda x: parse_case_day_slice(x)[0])

    # ============== 第一步：case_day 级别 train/val 划分 ==============
    all_case_days = sorted(df["case_day"].unique())
    random.shuffle(all_case_days)

    n_train_cases = int(len(all_case_days) * CFG.train_case_ratio)
    train_cases = set(all_case_days[:n_train_cases])
    val_cases   = set(all_case_days[n_train_cases:])

    # 先取出“初始训练集”和“验证集”的行
    df_train_all = df[df["case_day"].isin(train_cases)].reset_index(drop=True)
    df_val       = df[df["case_day"].isin(val_cases)].reset_index(drop=True)

    # ============== 第二步：在训练集中按 slice(id) 划分空/非空 ==============
    # 对于同一个 id，看这一张 slice 上是否“至少有一行有 mask”
    slice_has_mask = df_train_all.groupby("id")["has_mask_row"].any()

    pos_ids_train = [img_id for img_id, flag in slice_has_mask.items() if flag]      # 非空 slice
    neg_ids_train = [img_id for img_id, flag in slice_has_mask.items() if not flag]  # 空 slice

    print(f"Train slice ids (initial): {len(slice_has_mask)}")
    print(f"  ├─ non-empty slices (pos): {len(pos_ids_train)}")
    print(f"  └─ empty slices     (neg): {len(neg_ids_train)}")

    # 在训练集中下采样一部分“空 slice”
    if CFG.neg_keep_ratio < 1.0 and len(neg_ids_train) > 0:
        random.shuffle(neg_ids_train)
        n_keep_neg = max(1, int(len(neg_ids_train) * CFG.neg_keep_ratio))
        neg_ids_keep = neg_ids_train[:n_keep_neg]
    else:
        neg_ids_keep = neg_ids_train

    final_train_ids = set(pos_ids_train + neg_ids_keep)

    # 只保留这些 id 对应的行（所有类的行都会保留）
    df_train = df_train_all[df_train_all["id"].isin(final_train_ids)].reset_index(drop=True)

    print(f"Train slice ids (final): {len(final_train_ids)} "
          f"(pos={len(pos_ids_train)}, neg={len(neg_ids_keep)})")
    print(f"Val slice ids:         {df_val['id'].nunique()}")

    # ============== transforms & Dataset / DataLoader 保持原样 ==============
    train_transforms = A.Compose([
        A.Resize(CFG.img_size, CFG.img_size),
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.2),
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=10, p=0.2),
    ])

    val_transforms = A.Compose([
        A.Resize(CFG.img_size, CFG.img_size),
    ])

    train_dataset = UWGITract2_5D_Dataset(df_train, transforms=train_transforms)
    val_dataset   = UWGITract2_5D_Dataset(df_val,   transforms=val_transforms)

    print("Train samples (dataset len):", len(train_dataset))
    print("Val samples   (dataset len):", len(val_dataset))

    train_loader = DataLoader(
        train_dataset,
        batch_size=CFG.batch_size,
        shuffle=True,
        num_workers=CFG.num_workers,
        pin_memory=True,
        prefetch_factor=2,
        persistent_workers=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=1,   # 你原来怎么设就怎么来
        shuffle=False,
        num_workers=CFG.num_workers,
        pin_memory=True,
        prefetch_factor=2,
        persistent_workers=True,
    )

    return train_dataset, val_dataset, train_loader, val_loader

def debug_dataloader(loader, max_batches=3):
    """小测试：看看 DataLoader 能不能正常取 batch。"""
    print("len(loader):", len(loader))
    for i, (imgs, masks, img_ids) in enumerate(loader):
        print(f"batch {i}, imgs shape = {imgs.shape}, masks shape = {masks.shape}, first id = {img_ids[0]}")
        if i + 1 >= max_batches:
            break
# ================== TransUNet & losses ==================
class DiceLossMultiChannel(nn.Module):
    """对 3 通道二值 mask 的 Dice Loss。"""

    def __init__(self, smooth=1e-5):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred, target):
        pred = torch.sigmoid(pred)
        dims = (0, 2, 3)

        intersection = torch.sum(pred * target.float(), dims)
        pred_sum = torch.sum(pred * pred, dims)
        target_sum = torch.sum(target.float() * target.float(), dims)

        dice = (2 * intersection + self.smooth) / (pred_sum + target_sum + self.smooth)
        return 1 - dice.mean()


def build_transunet_model():
    """根据 CFG.vit_name 构建 TransUNet，并按需要加载预训练。"""
    # 1) 读取 backbone 配置（patch_size + 预训练路径）
    backbone_cfg = VIT_BACKBONES[CFG.vit_name]
    CFG.vit_patches_size = backbone_cfg["patch_size"]

    # 2) 从官方 CONFIGS 取对应的 ViT 配置
    config_vit = CONFIGS_ViT_seg[CFG.vit_name]
    config_vit.n_classes = CFG.num_classes
    config_vit.n_skip = CFG.n_skip

    # patch 大小
    config_vit.patches.size = (CFG.vit_patches_size, CFG.vit_patches_size)

    # 带 ResNet 的版本需要设置 grid（R50 / R26 都算）
    if ("R50" in CFG.vit_name) or ("R26" in CFG.vit_name):
        config_vit.patches.grid = (
            CFG.img_size // CFG.vit_patches_size,
            CFG.img_size // CFG.vit_patches_size,
        )

    # 预训练 npz 路径，从表里读取
    config_vit.pretrained_path = backbone_cfg["pretrained_path"]

    # 3) 构建模型
    model = ViT_seg(
        config_vit,
        img_size=CFG.img_size,
        num_classes=CFG.num_classes,
    )

    # 4) 按开关决定是否加载预训练
    if CFG.use_pretrained and config_vit.pretrained_path:
        print("尝试加载预训练权重:", config_vit.pretrained_path)
        if os.path.exists(config_vit.pretrained_path):
            weights = np.load(config_vit.pretrained_path)
            model.load_from(weights=weights)
            print("预训练权重加载成功。")
        else:
            print("[Warning] 找不到 npz，随机初始化。")
    else:
        print("不加载预训练，使用随机初始化。")

    return model.to(CFG.device)


# ================== train_one_epoch & evaluate_2d ==================
from tqdm.auto import tqdm


def train_one_epoch(epoch: int,scaler):
    model.train()
    running_loss = 0.0

    pbar = tqdm(enumerate(train_loader), total=len(train_loader))
    for step, (imgs, masks, _) in pbar:
        imgs  = imgs.to(CFG.device, non_blocking=True)
        masks = masks.to(CFG.device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        # ★ 1) 在 autocast 里做前向 & loss 计算
        with autocast(enabled=CFG.use_amp):
            logits = model(imgs)

            # >>> 新增：如果输出空间尺寸和 mask 不一致，就插值到 mask 尺寸 <<<
            if logits.shape[2:] != masks.shape[2:]:
                logits = F.interpolate(
                    logits,
                    size=masks.shape[2:],  # (H, W) = (256, 256)
                    mode="bilinear",
                    align_corners=False,
                )

            loss_bce = bce_loss_fn(logits, masks.float())
            loss_dice = dice_loss_fn(logits, masks)
            loss = 0.5 * loss_bce + 0.5 * loss_dice

        # ★ 2) 根据是否使用 AMP，决定如何 backward + step
        if CFG.use_amp:
            # 混合精度：先 scale，再 backward，再 step，再 update
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            # 传统 FP32：直接 backward + step
            loss.backward()
            optimizer.step()

        running_loss += loss.item()

        pbar.set_description(
            f"Epoch {epoch+1}/{CFG.num_epochs} "
            f"bce={loss_bce.item():.4f}, dice={loss_dice.item():.3f}, loss={loss.item():.3f}"
        )

    return running_loss / len(train_loader)

@torch.no_grad()
def evaluate_2d():
    model.eval()
    dices_all = []
    dices_pos = []

    with torch.no_grad():
        pbar = tqdm(enumerate(val_loader), total=len(val_loader), desc="Validating 2D")
        for step, (imgs, masks, _) in pbar:
            imgs  = imgs.to(CFG.device)
            masks = masks.to(CFG.device)

            logits = model(imgs)

            # >>> 新增：保证 logits 和 masks 的空间尺寸一致 <<<
            if logits.shape[2:] != masks.shape[2:]:
                logits = F.interpolate(
                    logits,
                    size=masks.shape[2:],
                    mode="bilinear",
                    align_corners=False,
                )

            probs = torch.sigmoid(logits).detach().cpu().numpy()
            gts = masks.detach().cpu().numpy()

            bs = imgs.size(0)
            for b in range(bs):
                p = probs[b]   # [C,H,W]
                g = gts[b]     # [C,H,W]
                for c in range(CFG.num_classes):
                    pred_bin = p[c] > 0.5
                    gt_bin   = g[c] > 0.5

                    # 所有 slice 的 Dice
                    d_all = compute_dice_2d_all(pred_bin, gt_bin)
                    dices_all.append(d_all)

                    # positive Dice
                    d_pos = compute_dice_2d_pos(pred_bin, gt_bin)
                    if d_pos is not None:
                        dices_pos.append(d_pos)

    dice_all = float(np.mean(dices_all)) if len(dices_all) > 0 else 0.0
    dice_pos = float(np.mean(dices_pos)) if len(dices_pos) > 0 else 0.0

    print(f"[Val 2D] Dice_all={dice_all:.4f}, Dice_pos={dice_pos:.4f}")
    return dice_all, dice_pos


# ================== helpers for 3D Dice & Hausdorff ==================
try:
    from medpy.metric.binary import hd95
except ImportError:
    hd95 = None
    print("[Warning] medpy 未安装，3D Hausdorff 将不可用。请先: pip install medpy")


def build_volume_maps_for_dataset(use_best_ckpt=True):
    """对 val_dataset 的每个样本做推理，收集成 3D 体。"""
    if use_best_ckpt and best_ckpt_path is not None and os.path.exists(best_ckpt_path):
        print("加载最佳模型权重用于 3D 评估:", best_ckpt_path)
        model.load_state_dict(torch.load(best_ckpt_path, map_location=CFG.device))

    model.eval()

    vol_pred = {}  # (case_day, c) -> list[(slice_num, 2D mask)]
    vol_gt   = {}

    with torch.no_grad():
        for idx in tqdm(range(len(val_dataset)), desc="Building 3D volumes"):
            imgs, masks, img_id = val_dataset[idx]
            imgs = imgs.unsqueeze(0).to(CFG.device)
            masks_np = masks.numpy()

            logits = model(imgs)

            # >>> 新增：对齐到 GT 的空间尺寸 <<<
            if logits.shape[2:] != masks_np.shape[-2:]:
                logits = F.interpolate(
                    logits,
                    size=masks_np.shape[-2:],  # (H, W)
                    mode="bilinear",
                    align_corners=False,
                )

            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).float().cpu().numpy()[0]

            case_day, slice_num = parse_case_day_slice(img_id)

            for c in range(CFG.num_classes):
                key = (case_day, c)
                vol_pred.setdefault(key, []).append((slice_num, preds[c] > 0.5))
                vol_gt.setdefault(key, []).append((slice_num, masks_np[c] > 0))

    return vol_pred, vol_gt

# ==== 3D Dice & HD95 helpers（替换原有 compute_3d_dice / compute_3d_hd95 / evaluate_3d）====

def compute_3d_dice_all(pred_vol, gt_vol):
    """
    3D Dice（all 版本）：
      - pred_vol, gt_vol: 3D numpy/bool 数组 [Z, H, W]
      - pred, gt 都空 => 1.0
      - 只有一边有前景 => 0.0
    """
    pred = pred_vol.astype(bool)
    gt   = gt_vol.astype(bool)

    pred_sum = np.sum(pred)
    gt_sum   = np.sum(gt)

    if pred_sum == 0 and gt_sum == 0:
        return 1.0
    if pred_sum == 0 or gt_sum == 0:
        return 0.0

    inter = np.logical_and(pred, gt).sum()
    return (2.0 * inter) / (pred_sum + gt_sum + 1e-8)


def compute_3d_dice_pos(pred_vol, gt_vol):
    """
    3D Dice（positive 版本）：
      - 只在 GT 有器官时参与平均；
      - GT=0 & pred=0 => 返回 None（忽略）
      - GT=0 & pred>0 => 误检，记 0
      - GT>0 & pred=0 => 漏检，记 0
    """
    pred = pred_vol.astype(bool)
    gt   = gt_vol.astype(bool)

    pred_sum = np.sum(pred)
    gt_sum   = np.sum(gt)

    # GT 本来就没有前景
    if gt_sum == 0:
        if pred_sum == 0:
            return None    # 完全背景，不参与平均
        else:
            return 0.0     # 纯误检，记 0

    # GT 有器官但预测全空 => 漏检
    if pred_sum == 0:
        return 0.0

    inter = np.logical_and(pred, gt).sum()
    return (2.0 * inter) / (pred_sum + gt_sum + 1e-8)


# 兼容一下之前如果还有地方用了 compute_3d_dice 的话
def compute_3d_dice(pred_vol, gt_vol):
    return compute_3d_dice_all(pred_vol, gt_vol)


def compute_3d_hd95(pred_vol, gt_vol):
    """
    3D 95% Hausdorff 距离：
      - 返回 None 表示这一对不计入均值（例如一边有前景、一边完全空）
      - pred, gt 都空 => 0.0
    """
    if hd95 is None:
        return None

    pred = pred_vol.astype(bool)
    gt   = gt_vol.astype(bool)

    pred_sum = np.sum(pred)
    gt_sum   = np.sum(gt)

    if pred_sum == 0 and gt_sum == 0:
        return 0.0
    if pred_sum == 0 or gt_sum == 0:
        return None

    return float(hd95(pred, gt))

def evaluate_3d():
    # 用 best ckpt 做 3D 评估
    vol_pred, vol_gt = build_volume_maps_for_dataset(use_best_ckpt=True)

    # 每个类别两套 Dice：all / pos
    dices_3d_all = {c: [] for c in range(CFG.num_classes)}
    dices_3d_pos = {c: [] for c in range(CFG.num_classes)}
    # HD95 和之前一样
    hd95s_3d     = {c: [] for c in range(CFG.num_classes) if hd95 is not None}

    for key, pred_list in vol_pred.items():
        case_day, c = key
        gt_list = vol_gt.get(key, [])

        if len(pred_list) == 0 or len(gt_list) == 0:
            continue

        # 按 slice_num 排序，堆成 3D 体 [Z, H, W]
        pred_list_sorted = sorted(pred_list, key=lambda x: x[0])
        gt_list_sorted   = sorted(gt_list,   key=lambda x: x[0])

        pred_vol = np.stack([mask for _, mask in pred_list_sorted], axis=0)
        gt_vol   = np.stack([mask for _, mask in gt_list_sorted],   axis=0)

        # 3D Dice（all）
        d_all = compute_3d_dice_all(pred_vol, gt_vol)
        dices_3d_all[c].append(d_all)

        # 3D Dice（pos）
        d_pos = compute_3d_dice_pos(pred_vol, gt_vol)
        if d_pos is not None:
            dices_3d_pos[c].append(d_pos)

        # HD95（逻辑不变）
        if hd95 is not None:
            h = compute_3d_hd95(pred_vol, gt_vol)
            if h is not None:
                hd95s_3d[c].append(h)

    # ===== 汇总并打印 =====
    print("====== 3D Volume-level Metrics ======")
    all_dice_all = []
    all_dice_pos = []
    all_hd       = []

    for c in range(CFG.num_classes):
        cls_name = IDX2CLASS[c]

        if len(dices_3d_all[c]) > 0:
            d_all_mean = float(np.mean(dices_3d_all[c]))
            all_dice_all.append(d_all_mean)
        else:
            d_all_mean = float("nan")

        if len(dices_3d_pos[c]) > 0:
            d_pos_mean = float(np.mean(dices_3d_pos[c]))
            all_dice_pos.append(d_pos_mean)
        else:
            d_pos_mean = float("nan")

        if hd95 is not None and c in hd95s_3d and len(hd95s_3d[c]) > 0:
            h_mean = float(np.mean(hd95s_3d[c]))
            all_hd.append(h_mean)
        else:
            h_mean = float("nan")

        print(
            f"Class {c} ({cls_name}):  "
            f"Dice3D_all={d_all_mean:.4f},  "
            f"Dice3D_pos={d_pos_mean:.4f},  "
            f"HD95={h_mean}"
        )

    print("-------------------------------------")
    if len(all_dice_all) > 0:
        print(f"Mean Dice3D_all (over classes): {np.mean(all_dice_all):.4f}")
    if len(all_dice_pos) > 0:
        print(f"Mean Dice3D_pos (over classes): {np.mean(all_dice_pos):.4f}")
    if hd95 is not None and len(all_hd) > 0:
        print(f"Mean HD95      (over classes): {np.mean(all_hd):.4f}")


# ================== visualization on val set ==================
@torch.no_grad()

def save_and_plot_history(history: dict):
    os.makedirs(CFG.out_dir, exist_ok=True)

    df_hist = pd.DataFrame(history)
    csv_path = os.path.join(CFG.out_dir, "train_history.csv")
    df_hist.to_csv(csv_path, index=False)
    print("训练日志已保存:", csv_path)

    plt.figure(figsize=(8, 5))
    plt.plot(df_hist["epoch"], df_hist["train_loss"], marker="o", label="Train Loss")

    # 如果有对应列，就分别画出来
    if "val_dice_all" in df_hist.columns:
        plt.plot(df_hist["epoch"], df_hist["val_dice_all"],
                 marker="s", label="Val Dice (all)")
    if "val_dice_pos" in df_hist.columns:
        plt.plot(df_hist["epoch"], df_hist["val_dice_pos"],
                 marker="^", label="Val Dice (pos)")

    plt.xlabel("Epoch")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.title("Training Loss & Validation Dice")

    png_path = os.path.join(CFG.out_dir, "loss_dice_curves.png")
    plt.savefig(png_path, dpi=150)
    plt.close()
    print("曲线图已保存:", png_path)


def visualize_val_samples(num_cases=5):
    model.eval()
    os.makedirs(os.path.join(CFG.out_dir, "vis"), exist_ok=True)

    indices = random.sample(range(len(val_dataset)), min(num_cases, len(val_dataset)))

    for i, idx in enumerate(indices):
        imgs, masks, img_id = val_dataset[idx]
        imgs_t = imgs.unsqueeze(0).to(CFG.device)
        masks_np = masks.numpy()          # [C, H, W]

        with torch.no_grad():
            logits = model(imgs_t)

            # ★ 新增：如果输出空间尺寸和 GT mask 不一致，就插值到 mask 尺寸
            # logits: [1, C, H_out, W_out]
            # masks_np: [C, H_gt, W_gt]
            if logits.shape[2:] != masks_np.shape[1:]:
                logits = F.interpolate(
                    logits,
                    size=masks_np.shape[1:],  # (H_gt, W_gt) = (256, 256)
                    mode="bilinear",
                    align_corners=False,
                )

            probs = torch.sigmoid(logits)

        preds = (probs > 0.5).float().cpu().numpy()[0]  # [C, H_gt, W_gt]

        # 取中间那张切片作为展示背景
        center_img = imgs[1].numpy()
        center_disp = enhance_image_natural(center_img)
        center_disp = np.stack([center_disp] * 3, axis=-1)   # [H, W, 3]

        gt_mask = masks_np
        pr_mask = preds

        # === 计算 per-class Dice: all / pos 两个版本 ===
        dices_all = []
        dices_pos = []

        for c in range(CFG.num_classes):
            pred_bin = pr_mask[c] > 0.5
            gt_bin   = gt_mask[c] > 0.5

            d_all = compute_dice_2d_all(pred_bin, gt_bin)
            d_pos = compute_dice_2d_pos(pred_bin, gt_bin)

            dices_all.append(d_all)
            dices_pos.append(d_pos)

        # mean Dice_all：所有通道都参与
        mean_dice_all = float(np.mean(dices_all)) if len(dices_all) > 0 else 0.0

        # mean Dice_pos：只对 GT 有器官 / 或误检的通道求平均（忽略 None）
        valid_pos = [d for d in dices_pos if d is not None]
        if len(valid_pos) > 0:
            mean_dice_pos = float(np.mean(valid_pos))
            mean_dice_pos_str = f"{mean_dice_pos:.3f}"
        else:
            mean_dice_pos = None
            mean_dice_pos_str = "NA"

        fig, axes = plt.subplots(1, 3, figsize=(18, 8))

        # ---- 左：原图 ----
        axes[0].imshow(center_disp, cmap="gray")
        axes[0].set_title(f"{img_id}\nCenter Slice", fontsize=14)
        axes[0].axis("off")

        # ---- 中/右：叠加 mask ----
        def overlay_mask(ax, bg_img, mask_3c, title):
            overlay = np.zeros_like(bg_img, dtype=np.float32)
            for ch in range(CFG.num_classes):
                m = mask_3c[ch]
                if np.sum(m) > 0:
                    color_layer = np.zeros_like(bg_img, dtype=np.float32)
                    color_layer[:, :, 0] = COLORS[ch][0] * m
                    color_layer[:, :, 1] = COLORS[ch][1] * m
                    color_layer[:, :, 2] = COLORS[ch][2] * m
                    overlay += color_layer * 0.5

            final_img = np.where(
                np.sum(overlay, axis=-1, keepdims=True) > 0.1,
                bg_img * 0.6 + overlay,
                bg_img
            )
            # 限制到 [0,1]，保持 float 显示
            final_img = np.clip(final_img, 0.0, 1.0)
            ax.imshow(final_img)
            ax.set_title(title, fontsize=14)
            ax.axis("off")

        overlay_mask(axes[1], center_disp, gt_mask, "Ground Truth")

        # 每类的字符串：C0:a=..,p=..  其中 p=None 的写成 NA
        per_cls_str = " ".join([
            f"C{c}:a={d_all:.2f},p={('NA' if d_pos is None else f'{d_pos:.2f}')}"
            for c, (d_all, d_pos) in enumerate(zip(dices_all, dices_pos))
        ])

        overlay_mask(
            axes[2],
            center_disp,
            pr_mask,
            (
                "Prediction\n"
                f"Mean Dice_all={mean_dice_all:.3f}, pos={mean_dice_pos_str}\n"
                + per_cls_str
            )
        )

        plt.tight_layout()
        save_path = os.path.join(CFG.out_dir, "vis", f"{img_id}_vis.png")
        plt.savefig(save_path, dpi=150)
        plt.close(fig)
        print(f"[{i+1}/{len(indices)}] Saved:", save_path)


# ================== main ==================
def main():
    global model, train_loader, val_loader, val_dataset, dice_loss_fn, bce_loss_fn, optimizer, best_ckpt_path

    # 准备数据
    train_dataset, val_dataset_, train_loader_, val_loader_ = create_datasets_and_loaders()
    val_dataset = val_dataset_
    train_loader = train_loader_
    val_loader = val_loader_

    # 构建模型与优化器
    model = build_transunet_model()
    print("Params (M):", sum(p.numel() for p in model.parameters()) / 1e6)

    dice_loss_fn = DiceLossMultiChannel()
    bce_loss_fn  = nn.BCEWithLogitsLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=CFG.lr,
        weight_decay=CFG.weight_decay,
    )

    scaler = GradScaler(enabled=CFG.use_amp)
    if CFG.use_scheduler:
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=CFG.lr_step_size,
            gamma=CFG.lr_gamma,
        )
    else:
        scheduler = None


    print("CFG.device =", CFG.device)
    print("torch.cuda.is_available() =", torch.cuda.is_available())
    print("model first param device =", next(model.parameters()).device)

    # 如果想测试 DataLoader，可以打开这一行：
    debug_dataloader(train_loader, max_batches=3)

    best_val_dice_2d = 0.0
    best_ckpt_path = os.path.join(CFG.out_dir, "best_transunet_2_5D.pth")

    # ★ 新增：用列表记录每个 epoch 的指标
    history = {
        "epoch": [],
        "train_loss": [],
        "val_dice_all": [],
        "val_dice_pos": [],
    }

    for epoch in range(CFG.num_epochs):
        train_loss = train_one_epoch(epoch, scaler)
        val_dice_all, val_dice_pos = evaluate_2d()

        print(f"==> Epoch [{epoch + 1}/{CFG.num_epochs}] "
              f"TrainLoss={train_loss:.4f}, "
              f"ValDice2D_all={val_dice_all:.4f}, ValDice2D_pos={val_dice_pos:.4f}")

        # --- 学习率调度：每个 epoch 结束后调用 ---
        if CFG.use_scheduler and scheduler is not None:
            scheduler.step()
            cur_lr = scheduler.get_last_lr()[0]
            print(f"  ↳ LR after scheduler: {cur_lr:.6f}")


        # --- 先把本轮结果全部记下来 ---
        history["epoch"].append(epoch + 1)
        history["train_loss"].append(train_loss)
        history["val_dice_all"].append(val_dice_all)
        history["val_dice_pos"].append(val_dice_pos)

        # --- 再看要不要刷新 best ---
        monitor = val_dice_pos  # 用 positive Dice 做 early stopping 指标
        if monitor > best_val_dice_2d:
            best_val_dice_2d = monitor
            torch.save(model.state_dict(), best_ckpt_path)
            print(f"  ↳ New best 2D Dice(pos): {best_val_dice_2d:.4f}, "
                  f"saved to {best_ckpt_path}")

    # for 循环结束后，再统一存日志 + 画图
    save_and_plot_history(history)

    # 然后再做 3D 评价 + 可视化
    evaluate_3d()
    visualize_val_samples(num_cases=5)


if __name__ == "__main__":
    main()
