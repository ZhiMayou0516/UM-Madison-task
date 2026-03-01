import os
import cv2
import glob
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
import segmentation_models_pytorch as smp
from tqdm import tqdm
from sklearn.model_selection import GroupKFold
import importlib

class Config:
    DATA_ROOT = "/data/lxy/ML25/"
    MODEL_NAME = 'LeViT_UNet_384' 
    
    SEED = 42
    img_size = 224
    batch_size = 8 
    epochs = 50
    lr = 1e-4
    num_classes = 3
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    save_best_name = f"best_model_{MODEL_NAME}.pth"
    checkpoint_name = f"checkpoint_{MODEL_NAME}.pth"

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
    
    def extract_id(path):
        parts = path.replace('\\', '/').split('/')
        if 'scans' in parts[-2]: case_day = parts[-3]
        else: case_day = parts[-2]
        slice_id = parts[-1].split('_')[1]
        return f"{case_day}_slice_{slice_id}"

    path_df['id'] = path_df['image_path'].apply(extract_id)
    path_df['case_id'] = path_df['id'].apply(lambda x: x.split('_')[0])
    
    df = pd.read_csv(os.path.join(data_root, "train.csv"))
    df.fillna('', inplace=True)
    df_pivot = df.pivot(index='id', columns='class', values='segmentation').reset_index()
    
    return pd.merge(path_df, df_pivot, on='id', how='inner')

class LeViTDataset(Dataset):
    def __init__(self, df, transforms=None):
        self.df = df
        self.transforms = transforms
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = row['image_path']
        
        try:
            img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        except Exception as e:
            print(f"Warning:  {img_path}, Error: {e}")
            img = None
            
        if img is None:
            img = np.zeros((Config.img_size, Config.img_size), dtype=np.uint16)
        
        h, w = img.shape
        
        img = img.astype(np.float32)
        min_val, max_val = img.min(), img.max()
        if max_val > min_val:
            img = (img - min_val) / (max_val - min_val)
        else:
            img = img / 65535.0
            
        img = np.expand_dims(img, axis=-1) # (H, W, 1)
        
        masks = []
        for cls in ['large_bowel', 'small_bowel', 'stomach']:
            masks.append(rle_decode(row[cls], (h, w)))
        mask = np.stack(masks, axis=-1).astype(np.float32)
        
        if self.transforms:
            aug = self.transforms(image=img, mask=mask)
            img = aug['image']
            mask = aug['mask']
            mask = mask.permute(2, 0, 1)
            
        return img, mask

data_transforms = {
    "train": A.Compose([A.Resize(Config.img_size, Config.img_size), A.HorizontalFlip(p=0.5), A.VerticalFlip(p=0.5), ToTensorV2()]),
    "valid": A.Compose([A.Resize(Config.img_size, Config.img_size), ToTensorV2()])
}

def build_levit_model():
    print(f"Loading Model from file: {Config.MODEL_NAME}.py ...")
    try:
        module = importlib.import_module(Config.MODEL_NAME)
    except ImportError as e:
        print(f"Error: Can't find  {Config.MODEL_NAME}.py")
        raise e
    
    if '128s' in Config.MODEL_NAME: func_name = 'Build_LeViT_UNet_128s'
    elif '192' in Config.MODEL_NAME: func_name = 'Build_LeViT_UNet_192'
    elif '384' in Config.MODEL_NAME: func_name = 'Build_LeViT_UNet_384'
    else: raise ValueError("Unknown LeViT model type")
        
    build_func = getattr(module, func_name)
    model = build_func(num_classes=Config.num_classes, pretrained=False, fuse=False)
    return model

def train():
    print("Preparing Data...")
    df = get_metadata(Config.DATA_ROOT)
    gkf = GroupKFold(n_splits=10)
    for train_idx, valid_idx in gkf.split(df, groups=df['case_id']):
        train_df = df.iloc[train_idx]
        valid_df = df.iloc[valid_idx]
        break
    
    train_loader = DataLoader(LeViTDataset(train_df, data_transforms['train']), batch_size=Config.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    valid_loader = DataLoader(LeViTDataset(valid_df, data_transforms['valid']), batch_size=Config.batch_size, shuffle=False, num_workers=4, pin_memory=True)
    
    model = build_levit_model()
    model.to(Config.device)
    optimizer = optim.AdamW(model.parameters(), lr=Config.lr, weight_decay=1e-4)
    
    dice_loss = smp.losses.DiceLoss(mode='multilabel')
    bce_loss = smp.losses.SoftBCEWithLogitsLoss()
    def criterion(pred, target): return 0.5 * dice_loss(pred, target) + 0.5 * bce_loss(pred, target)
    
    start_epoch = 0
    best_loss = float('inf')
    
    if os.path.exists(Config.checkpoint_name):
        try:
            checkpoint = torch.load(Config.checkpoint_name, map_location=Config.device)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            best_loss = checkpoint.get('best_loss', float('inf'))
        except Exception as e:
            print(f" {e}")
    else:
        print("No checkpoint found, training from scratch.")
        
    print(f"Start Training {Config.MODEL_NAME}...")
    for epoch in range(start_epoch, Config.epochs):
        model.train()
        train_loss = 0
        pbar = tqdm(train_loader, desc=f"Ep {epoch+1}/{Config.epochs}")
        
        for imgs, masks in pbar:
            imgs, masks = imgs.to(Config.device), masks.to(Config.device)
            optimizer.zero_grad()
            outputs = model(imgs)
            
            if outputs.shape[-2:] != masks.shape[-2:]:
                outputs = nn.functional.interpolate(outputs, size=masks.shape[-2:], mode='bilinear', align_corners=False)
            
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            pbar.set_postfix(loss=loss.item())
            
        model.eval()
        valid_loss = 0
        with torch.no_grad():
            for imgs, masks in valid_loader:
                imgs, masks = imgs.to(Config.device), masks.to(Config.device)
                outputs = model(imgs)
                if outputs.shape[-2:] != masks.shape[-2:]:
                    outputs = nn.functional.interpolate(outputs, size=masks.shape[-2:], mode='bilinear', align_corners=False)
                loss = criterion(outputs, masks)
                valid_loss += loss.item()
                
        avg_train_loss = train_loss / len(train_loader)
        avg_valid_loss = valid_loss / len(valid_loader)
        print(f"Train Loss: {avg_train_loss:.4f} | Valid Loss: {avg_valid_loss:.4f}")
        
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_loss': best_loss
        }, Config.checkpoint_name)
        
        if avg_valid_loss < best_loss:
            best_loss = avg_valid_loss
            torch.save(model.state_dict(), Config.save_best_name)
            print(f">>> Best Model Updated: {best_loss:.4f}")

if __name__ == '__main__':
    train()