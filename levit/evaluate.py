import torch
import numpy as np
import pandas as pd
import os
import glob
import cv2
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
import segmentation_models_pytorch as smp
from sklearn.model_selection import GroupKFold
import importlib

class Config:
    DATA_ROOT = "/data/lxy/ML25/"
    
    # DeepLabV3+
    MODEL_TYPE = 'deeplab'
    ENCODER = 'efficientnet-b4'
    MODEL_PATH = "/data/lxy/ML25/weights/best_model_deeplabv3plus_efficientnet-b4.pth"
    IN_CHANNELS = 3  
    IMG_SIZE = 256
    
    # LeViT-128s 
    #MODEL_TYPE = 'LeViT_UNet_384' 
    #MODEL_PATH = "weights/best_model_LeViT_UNet_384.pth"
    #IN_CHANNELS = 1          
    #IMG_SIZE = 224         
    
    BATCH_SIZE = 16
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    SEED = 42

def rle_decode(mask_rle, shape):
    if str(mask_rle) == 'nan' or mask_rle == '': return np.zeros(shape, dtype=np.uint8)
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0]*shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape)

def get_metadata(data_root):
    train_images = glob.glob(os.path.join(data_root, "train", "**", "*.png"), recursive=True)
    path_df = pd.DataFrame(train_images, columns=['image_path'])
    path_df['id'] = path_df['image_path'].apply(lambda x: f"{x.split('/')[-3] if 'scans' in x.split('/')[-2] else x.split('/')[-2]}_slice_{x.split('/')[-1].split('_')[1]}")
    df = pd.read_csv(os.path.join(data_root, "train.csv"))
    df.fillna('', inplace=True)
    df_pivot = df.pivot(index='id', columns='class', values='segmentation').reset_index()
    final_df = pd.merge(path_df, df_pivot, on='id', how='inner')
    final_df['case_id'] = final_df['id'].apply(lambda x: x.split('_')[0])
    return final_df

class UWDataset(Dataset):
    def __init__(self, df, transforms=None, in_channels=3):
        self.df = df
        self.transforms = transforms
        self.in_channels = in_channels
        
    def __len__(self): return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = cv2.imread(row['image_path'], cv2.IMREAD_UNCHANGED)
        h, w = img.shape
        
        img = img.astype(np.float32)
        min_val, max_val = img.min(), img.max()
        if max_val > min_val: img = (img - min_val) / (max_val - min_val)
        else: img = img / 65535.0
            
        if self.in_channels == 3:
            img = np.stack([img, img, img], axis=-1)
        else:
            img = np.expand_dims(img, axis=-1)
            
        masks = [rle_decode(row[c], (h, w)) for c in ['large_bowel', 'small_bowel', 'stomach']]
        mask = np.stack(masks, axis=-1).astype(np.float32)
        
        if self.transforms:
            aug = self.transforms(image=img, mask=mask)
            img = aug['image']
            mask = aug['mask'].permute(2, 0, 1)
            
        return img, mask

def compute_metrics_fixed(pred_mask, gt_mask):
    dice_scores, gt_exists = [], []
    eps = 1e-7
    for ch in range(3):
        pred, gt = pred_mask[ch], gt_mask[ch]
        gt_exists.append(np.sum(gt) > 0)
        if np.sum(gt) == 0 and np.sum(pred) == 0: dice = 1.0
        else: dice = (2. * np.sum(pred * gt) + eps) / (np.sum(pred) + np.sum(gt) + eps)
        dice_scores.append(dice)
    return dice_scores, gt_exists

def evaluate():
    df = get_metadata(Config.DATA_ROOT)
    gkf = GroupKFold(n_splits=10)
    for train_idx, valid_idx in gkf.split(df, groups=df['case_id']):
        valid_df = df.iloc[valid_idx]
        break
    
    ds = UWDataset(valid_df, 
                   transforms=A.Compose([A.Resize(Config.IMG_SIZE, Config.IMG_SIZE), ToTensorV2()]),
                   in_channels=Config.IN_CHANNELS)
    loader = DataLoader(ds, batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=4)
    
    if Config.MODEL_TYPE == 'deeplab':
        print(f"Loading DeepLabV3+ ({Config.ENCODER})...")
        model = smp.DeepLabV3Plus(encoder_name=Config.ENCODER, encoder_weights=None, in_channels=Config.IN_CHANNELS, classes=3)
    elif 'LeViT' in Config.MODEL_TYPE:
        print(f"Loading LeViT ({Config.MODEL_TYPE})...")
        module = importlib.import_module(Config.MODEL_TYPE)
        if '128s' in Config.MODEL_TYPE: func_name = 'Build_LeViT_UNet_128s'
        elif '192' in Config.MODEL_TYPE: func_name = 'Build_LeViT_UNet_192'
        elif '384' in Config.MODEL_TYPE: func_name = 'Build_LeViT_UNet_384'
        model = getattr(module, func_name)(num_classes=3, pretrained=False, fuse=False)
    
    print(f"Loading Weights: {Config.MODEL_PATH}")
    checkpoint = torch.load(Config.MODEL_PATH, map_location=Config.DEVICE)
    state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
    model.load_state_dict(state_dict)
    model.to(Config.DEVICE)
    model.eval()
    
    total_dice = np.zeros(3)
    pos_dice_sum = np.zeros(3)
    count_all = 0
    count_pos = np.zeros(3)
    
    print("Evaluating...")
    with torch.no_grad():
        for imgs, masks in tqdm(loader):
            imgs = imgs.to(Config.DEVICE)
            logits = model(imgs)
            
            if logits.shape[-2:] != masks.shape[-2:]:
                logits = torch.nn.functional.interpolate(logits, size=masks.shape[-2:], mode='bilinear')
                
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).float().cpu().numpy()
            masks = masks.numpy()
            
            for i in range(len(imgs)):
                dices, exists = compute_metrics_fixed(preds[i], masks[i])
                total_dice += np.array(dices)
                count_all += 1
                for ch in range(3):
                    if exists[ch]:
                        pos_dice_sum[ch] += dices[ch]
                        count_pos[ch] += 1
    
    avg_dice_all = total_dice / count_all
    avg_dice_pos = np.divide(pos_dice_sum, count_pos, out=np.zeros_like(pos_dice_sum), where=count_pos!=0)
    
    print("\n" + "="*60)
    print(f"Evaluation Report (Model: {Config.MODEL_TYPE})")
    print("-" * 60)
    print(f"{'Class':<15} | {'Mean Dice (All)':<20} | {'Mean Dice (Positive)':<25}")
    print(f"{'Large Bowel':<15} | {avg_dice_all[0]:.4f}               | {avg_dice_pos[0]:.4f}")
    print(f"{'Small Bowel':<15} | {avg_dice_all[1]:.4f}               | {avg_dice_pos[1]:.4f}")
    print(f"{'Stomach':<15} | {avg_dice_all[2]:.4f}               | {avg_dice_pos[2]:.4f}")
    print("-" * 60)
    print(f"{'Overall':<15} | {np.mean(avg_dice_all):.4f}               | {np.mean(avg_dice_pos):.4f}")
    print("="*60)

if __name__ == "__main__":
    evaluate()