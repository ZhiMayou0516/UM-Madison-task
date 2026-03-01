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
from medpy.metric.binary import hd95

class Config:
    DATA_ROOT = "/data/lxy/ML25/"
    
    #DeepLab
    MODEL_TYPE = 'deeplab'
    MODEL_PATH = "/data/lxy/ML25/weights/best_model_deeplabv3plus_mobilenet_v2.pth"
    ENCODER = 'mobilenet_v2'
    IN_CHANNELS = 3
    IMG_SIZE = 256
    
    #LeViT
    #MODEL_TYPE = 'LeViT_UNet_384'
    #MODEL_PATH = "weights/best_model_LeViT_UNet_384.pth"
    #ENCODER = 'LeViT-384'
    #IN_CHANNELS = 1
    #IMG_SIZE = 224
    
    BATCH_SIZE = 16
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    SEED = 42
    NUM_CLASSES = 3
    CLASSES = ['Large Bowel', 'Small Bowel', 'Stomach']

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
    
    def extract_info(path):
        parts = path.replace('\\', '/').split('/')
        if 'scans' in parts[-2]: case_day = parts[-3]
        else: case_day = parts[-2]
        slice_id = int(parts[-1].split('_')[1]) 
        full_id = f"{case_day}_slice_{slice_id:04d}"
        return full_id, case_day, slice_id

    infos = path_df['image_path'].apply(extract_info)
    path_df['id'] = [x[0] for x in infos]
    path_df['case_day'] = [x[1] for x in infos] 
    path_df['slice_num'] = [x[2] for x in infos] 
    path_df['case_id'] = path_df['case_day'].apply(lambda x: x.split('_')[0])
    
    df = pd.read_csv(os.path.join(data_root, "train.csv"))
    df.fillna('', inplace=True)
    df_pivot = df.pivot(index='id', columns='class', values='segmentation').reset_index()
    
    final_df = pd.merge(path_df, df_pivot, on='id', how='inner')
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
            
        if self.in_channels == 3: img = np.stack([img, img, img], axis=-1)
        else: img = np.expand_dims(img, axis=-1)
            
        masks = [rle_decode(row[c.lower().replace(' ', '_')], (h, w)) for c in Config.CLASSES]
        mask = np.stack(masks, axis=-1).astype(np.float32)
        
        if self.transforms:
            aug = self.transforms(image=img, mask=mask)
            img = aug['image']
            mask = aug['mask'].permute(2, 0, 1)
           
        return img, mask, row['case_day'], row['slice_num']

def compute_3d_hd95(pred_vol, gt_vol):
    if hd95 is None: return None
    
    pred = pred_vol.astype(bool)
    gt = gt_vol.astype(bool)
    
    if np.sum(pred) == 0 and np.sum(gt) == 0:
        return 0.0 
    
    if np.sum(pred) == 0 or np.sum(gt) == 0:
        return None 
        
    try:
        return hd95(pred, gt)
    except Exception as e:
        print(f"HD95: {e}")
        return None

def compute_3d_dice(pred_vol, gt_vol):
    pred = pred_vol.astype(bool)
    gt = gt_vol.astype(bool)
    
    if np.sum(pred) == 0 and np.sum(gt) == 0: return 1.0
    if np.sum(pred) == 0 or np.sum(gt) == 0: return 0.0
    
    intersection = np.logical_and(pred, gt).sum()
    return 2.0 * intersection / (pred.sum() + gt.sum() + 1e-8)

def evaluate_3d():
    if hd95 is None: return

    df = get_metadata(Config.DATA_ROOT)
    gkf = GroupKFold(n_splits=10)
    for train_idx, valid_idx in gkf.split(df, groups=df['case_id']):
        valid_df = df.iloc[valid_idx]
        break
    
    
    ds = UWDataset(valid_df, 
                   transforms=A.Compose([A.Resize(Config.IMG_SIZE, Config.IMG_SIZE), ToTensorV2()]),
                   in_channels=Config.IN_CHANNELS)
    loader = DataLoader(ds, batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=4)
    
    print(f"Loading Model: {Config.MODEL_TYPE}...")
    if Config.MODEL_TYPE == 'deeplab':
        model = smp.DeepLabV3Plus(encoder_name=Config.ENCODER, encoder_weights=None, in_channels=Config.IN_CHANNELS, classes=3)
    elif 'LeViT' in Config.MODEL_TYPE:
        module = importlib.import_module(Config.MODEL_TYPE)
        if '128s' in Config.MODEL_TYPE: func_name = 'Build_LeViT_UNet_128s'
        elif '192' in Config.MODEL_TYPE: func_name = 'Build_LeViT_UNet_192'
        elif '384' in Config.MODEL_TYPE: func_name = 'Build_LeViT_UNet_384'
        model = getattr(module, func_name)(num_classes=3, pretrained=False, fuse=False)
        
    try:
        checkpoint = torch.load(Config.MODEL_PATH, map_location=Config.DEVICE)
        state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
        model.load_state_dict(state_dict)
    except Exception as e:
        print(f"Error loading weights: {e}"); return

    model.to(Config.DEVICE)
    model.eval()
    vol_pred = {} 
    vol_gt = {}
    
    with torch.no_grad():
        for imgs, masks, case_days, slice_nums in tqdm(loader):
            imgs = imgs.to(Config.DEVICE)
            logits = model(imgs)
            
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).float().cpu().numpy() 
            gts = masks.numpy() 
            
            for i in range(len(imgs)):
                c_day = case_days[i]
                s_num = int(slice_nums[i])
                
                for c in range(3):
                    key = (c_day, c)
                    if key not in vol_pred: vol_pred[key] = []
                    if key not in vol_gt: vol_gt[key] = []
                    
                    vol_pred[key].append((s_num, preds[i][c]))
                    vol_gt[key].append((s_num, gts[i][c]))

    
    print("\n Calculating (Dice & HD95)...")
    
    metrics = {
        'Large Bowel': {'dice': [], 'hd95': []},
        'Small Bowel': {'dice': [], 'hd95': []},
        'Stomach':     {'dice': [], 'hd95': []}
    }
    
    unique_keys = list(vol_pred.keys())
    
    for key in tqdm(unique_keys):
        case_day, c_idx = key
        class_name = Config.CLASSES[c_idx]
        
        p_list = vol_pred[key]
        g_list = vol_gt[key]
        
        p_list.sort(key=lambda x: x[0])
        g_list.sort(key=lambda x: x[0])
        
        p_vol = np.stack([x[1] for x in p_list])
        g_vol = np.stack([x[1] for x in g_list])
        
        dice = compute_3d_dice(p_vol, g_vol)
        hd = compute_3d_hd95(p_vol, g_vol)
        
        if dice is not None: metrics[class_name]['dice'].append(dice)
        if hd is not None: metrics[class_name]['hd95'].append(hd)

    print("\n" + "="*50)
    print(f"3D Evaluate report (Model: {Config.MODEL_TYPE})")
    print("-" * 50)
    print(f"{'Class':<15} | {'3D Dice':<12} | {'3D HD95':<12}")
    print("-" * 50)
    
    avg_dice_all = []
    avg_hd_all = []
    
    for cls in Config.CLASSES:
        d_mean = np.mean(metrics[cls]['dice'])
        h_mean = np.mean(metrics[cls]['hd95']) if len(metrics[cls]['hd95']) > 0 else np.nan
        
        avg_dice_all.append(d_mean)
        if not np.isnan(h_mean): avg_hd_all.append(h_mean)
        
        print(f"{cls:<15} | {d_mean:.4f}       | {h_mean:.4f}")
        
    print("-" * 50)
    print(f"{'Mean':<15} | {np.mean(avg_dice_all):.4f}       | {np.mean(avg_hd_all):.4f}")
    print("="*50)

if __name__ == "__main__":
    evaluate_3d()