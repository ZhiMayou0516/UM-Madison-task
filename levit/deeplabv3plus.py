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
import matplotlib.pyplot as plt

class Config:
    DATA_ROOT = "/data/lxy/ML25/"
    
    checkpoint_path = "weights/checkpoint_deeplabv3plus_mobilenet_v2.pth"    
    best_model_path = "weights/best_model_deeplabv3plus_mobilenet_v2.pth"    
    
    SEED = 42
    img_size = (256, 256)   
    batch_size = 16         
    epochs = 50             
    lr = 1e-4            
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    encoder = 'mobilenet_v2'    
    weights = 'imagenet'
    num_classes = 3        

def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)

set_seed(Config.SEED)

def get_metadata(data_root):
    train_images = glob.glob(os.path.join(data_root, "train", "**", "*.png"), recursive=True)
    
    path_df = pd.DataFrame(train_images, columns=['image_path'])
    
    def extract_id(path):
        parts = path.replace('\\', '/').split('/')
        if 'scans' in parts[-2]:
            case_day = parts[-3]
        else:
            case_day = parts[-2]
            
        filename = parts[-1]
        slice_id = filename.split('_')[1] 
        return f"{case_day}_slice_{slice_id}"

    path_df['id'] = path_df['image_path'].apply(extract_id)
    
    df = pd.read_csv(os.path.join(data_root, "train.csv"))
    
    df.fillna('', inplace=True) 
    df_pivot = df.pivot(index='id', columns='class', values='segmentation').reset_index()
    
    final_df = pd.merge(path_df, df_pivot, on='id', how='inner')
    
    return final_df

def rle_decode(mask_rle, shape):
    if mask_rle == '':
        return np.zeros(shape, dtype=np.uint8)
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0]*shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape)

class UWDataset(Dataset):
    def __init__(self, df, transforms=None):
        self.df = df
        self.transforms = transforms
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = row['image_path']
        
        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        if img is None:
            raise FileNotFoundError(f"Cannot read img: {img_path}")
        h, w = img.shape[:2] 
        
        img = img.astype(np.float32)
        min_val = img.min()
        max_val = img.max()
        if max_val > min_val:
            img = (img - min_val) / (max_val - min_val)
        else:
            img = img / 65535.0
            
        img = np.stack([img, img, img], axis=-1)
        
        masks = []
        for class_name in ['large_bowel', 'small_bowel', 'stomach']:
            rle = row[class_name]
            mask = rle_decode(rle, shape=(h, w))
            masks.append(mask)
        mask = np.stack(masks, axis=-1).astype(np.float32)
        
        if self.transforms:
            try:
                augmented = self.transforms(image=img, mask=mask)
                img = augmented['image']
                mask = augmented['mask']
                mask = mask.permute(2, 0, 1) 
            except ValueError as e:
                print(f"Augmentation Error at index {idx}: Path: {img_path}")
                raise e
            
        return img, mask

data_transforms = {
    "train": A.Compose([
        A.Resize(*Config.img_size),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        ToTensorV2(),
    ]),
    "valid": A.Compose([
        A.Resize(*Config.img_size),
        ToTensorV2(),
    ])
}


def train_model():
    df = get_metadata(Config.DATA_ROOT)
    
    from sklearn.model_selection import GroupKFold
    df['case_id'] = df['id'].apply(lambda x: x.split('_')[0])
    gkf = GroupKFold(n_splits=10)
    for train_idx, valid_idx in gkf.split(df, groups=df['case_id']):
        train_df = df.iloc[train_idx]
        valid_df = df.iloc[valid_idx]
        break 
    
    train_dataset = UWDataset(train_df, transforms=data_transforms['train'])
    valid_dataset = UWDataset(valid_df, transforms=data_transforms['valid'])
    
    train_loader = DataLoader(train_dataset, batch_size=Config.batch_size, shuffle=True, num_workers=0, pin_memory=True)
    valid_loader = DataLoader(valid_dataset, batch_size=Config.batch_size, shuffle=False, num_workers=0, pin_memory=True)
    
    model = smp.DeepLabV3Plus(
        encoder_name=Config.encoder, 
        encoder_weights=None,
        in_channels=3,
        classes=Config.num_classes,
        activation=None, 
    )
    
    local_weight_path = "/data/lxy/ML25/pretrained_weights/mobilenet_v2-b0353104.pth"
    
    
    if os.path.exists(local_weight_path):
        print(f"Loading Backbone pretrained weights: {local_weight_path}")
        try:
            state_dict = torch.load(local_weight_path, weights_only=False)
            keys_to_remove = ["fc.weight", "fc.bias"]
            for key in keys_to_remove:
                if key in state_dict:
                    del state_dict[key]
            model.encoder.load_state_dict(state_dict) 
        except Exception as e:
            print(f"Backbone loading error: {e}")
    else:
        print(f"Error can't find Backbone  {local_weight_path}")
    
    model.to(Config.device)
    
    optimizer = optim.AdamW(model.parameters(), lr=Config.lr, weight_decay=1e-5)
    
    dice_loss = smp.losses.DiceLoss(mode='multilabel')
    bce_loss = smp.losses.SoftBCEWithLogitsLoss()
    
    def criterion(pred, target):
        return 0.5 * dice_loss(pred, target) + 0.5 * bce_loss(pred, target)
    
    start_epoch = 0
    best_loss = float('inf')
    
    if os.path.exists(Config.checkpoint_path):
        try:
            checkpoint = torch.load(Config.checkpoint_path, map_location=Config.device, weights_only=False)
            
            model.load_state_dict(checkpoint['model_state_dict'])
            
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
            start_epoch = checkpoint['epoch'] + 1
            best_loss = checkpoint.get('best_loss', float('inf'))
            
        except Exception as e:
            print(f"Error: {e}")
    else:
        print("\n Cannot find checkpoint")

    for epoch in range(start_epoch, Config.epochs):
        print(f"\nEpoch {epoch+1}/{Config.epochs}")
        
        model.train()
        train_loss = 0
        pbar = tqdm(train_loader, desc=f"Train Ep {epoch+1}")
        
        for imgs, masks in pbar:
            imgs = imgs.to(Config.device)
            masks = masks.to(Config.device)
            
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, masks)
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")
            
        train_epoch_loss = train_loss / len(train_loader)
        
        model.eval()
        valid_loss = 0
        with torch.no_grad():
            for imgs, masks in DataLoader(valid_dataset, batch_size=Config.batch_size):
                imgs = imgs.to(Config.device)
                masks = masks.to(Config.device)
                
                outputs = model(imgs)
                loss = criterion(outputs, masks)
                valid_loss += loss.item()
                
        valid_epoch_loss = valid_loss / len(valid_loader)
        
        print(f"Train Loss: {train_epoch_loss:.4f} | Valid Loss: {valid_epoch_loss:.4f}")
        
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_loss': best_loss,
        }, Config.checkpoint_path + ".tmp")
        
        if os.path.exists(Config.checkpoint_path + ".tmp"):
            os.replace(Config.checkpoint_path + ".tmp", Config.checkpoint_path)
            
        if valid_epoch_loss < best_loss:
            best_loss = valid_epoch_loss
            torch.save(model.state_dict(), Config.best_model_path)
            
        print("Checkpoint saved.")

    print("\nTraining Completed!")

if __name__ == '__main__':
    train_model()