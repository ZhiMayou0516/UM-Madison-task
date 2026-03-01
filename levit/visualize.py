import torch
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
import glob
import pandas as pd
import segmentation_models_pytorch as smp
import albumentations as A
from albumentations.pytorch import ToTensorV2
import importlib

class Config:
    # DeepLab (ResNet50)
    MODEL_TYPE = 'deeplab'
    MODEL_PATH = "/data/lxy/ML25/weights/best_model_deeplabv3plus_efficientnet-b4.pth" 
    ENCODER = 'efficientnet-b4'
    IN_CHANNELS = 3      
    IMG_SIZE = 256
    SAVE_DIR = "results_deeplabv3plus_efficientnet-b4"
    
    # LeViT (128s)
    #MODEL_TYPE = 'LeViT_UNet_384' 
    #MODEL_PATH = "weights/best_model_LeViT_UNet_384.pth"
    #ENCODER = 'LeViT-384' 
    #IN_CHANNELS = 1       
    #IMG_SIZE = 224     
    #SAVE_DIR = "results_levit_unet_384"
    
    DATA_ROOT = "/data/lxy/ML25/"
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

COLORS = [
    [0.9, 0.0, 0.0], # Large Bowel 
    [0.0, 0.8, 0.0], # Small Bowel
    [0.0, 0.0, 0.9], # Stomach 
]

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

def compute_dice(pred, gt):
    pred_sum = np.sum(pred)
    gt_sum = np.sum(gt)
    if gt_sum == 0 and pred_sum == 0: return 1.0
    intersection = np.sum(pred * gt)
    return (2. * intersection) / (pred_sum + gt_sum + 1e-8)

def enhance_image_natural(img_array):
    img = img_array.astype(np.float32)
    p_high = np.percentile(img, 99.5)
    img = np.clip(img, 0, p_high)
    img = (img - img.min()) / (img.max() - img.min() + 1e-8)
    return img

def find_diverse_samples(data_root, num_cases=5):
    train_images = glob.glob(os.path.join(data_root, "train", "**", "*.png"), recursive=True)
    path_map = {} 
    for path in train_images:
        parts = path.replace('\\', '/').split('/')
        if 'scans' in parts[-2]: case_day = parts[-3]
        else: case_day = parts[-2]
        filename = parts[-1]
        slice_id = filename.split('_')[1]
        img_id = f"{case_day}_slice_{slice_id}"
        path_map[img_id] = path

    df = pd.read_csv(os.path.join(data_root, "train.csv"))
    df.fillna('', inplace=True)
    df_valid = df[df['segmentation'] != ''].copy()
    df_valid['case_id'] = df_valid['id'].apply(lambda x: x.split('_')[0])
    unique_cases = df_valid['case_id'].unique()
    
    selected_samples = []
    count = 0
    for case in unique_cases:
        if count >= num_cases: break
        case_df = df_valid[df_valid['case_id'] == case]
        if len(case_df) > 0:
            sample = case_df.sample(1).iloc[0]
            img_path = path_map.get(sample['id'])
            if img_path:
                full_info = df[df['id'] == sample['id']]
                selected_samples.append({'id': sample['id'], 'path': img_path, 'df_rows': full_info})
                count += 1
    return selected_samples

def plot_mask_overlay(ax, bg_img, mask_3ch, title, is_pred=False):
    ax.imshow(bg_img)
    overlay = np.zeros_like(bg_img)
    for ch in range(3):
        m = mask_3ch[ch]
        if np.sum(m) > 0:
            color_layer = np.zeros_like(bg_img)
            color_layer[:,:,0] = COLORS[ch][0] * m
            color_layer[:,:,1] = COLORS[ch][1] * m
            color_layer[:,:,2] = COLORS[ch][2] * m
            overlay += color_layer * 0.5
    final_img = np.where(np.sum(overlay, axis=-1, keepdims=True) > 0.1, bg_img * 0.6 + overlay, bg_img)
    ax.imshow(final_img)
    ax.set_title(title, fontsize=14, pad=10)
    ax.axis('off')

def main():
    os.makedirs(Config.SAVE_DIR, exist_ok=True)
    
    print(f"Building Model: {Config.MODEL_TYPE}...")
    if Config.MODEL_TYPE == 'deeplab':
        model = smp.DeepLabV3Plus(
            encoder_name=Config.ENCODER,
            encoder_weights=None,
            in_channels=Config.IN_CHANNELS,
            classes=3,
            activation=None,
        )
    elif 'LeViT' in Config.MODEL_TYPE:
        try:
            module = importlib.import_module(Config.MODEL_TYPE) 
            if '128s' in Config.MODEL_TYPE: func_name = 'Build_LeViT_UNet_128s'
            elif '192' in Config.MODEL_TYPE: func_name = 'Build_LeViT_UNet_192'
            elif '384' in Config.MODEL_TYPE: func_name = 'Build_LeViT_UNet_384'
            model = getattr(module, func_name)(num_classes=3, pretrained=False, fuse=False)
        except Exception as e:
            print(f"Error loading LeViT module: {e}")
            return

    print(f"Loading Weights: {Config.MODEL_PATH}")
    try:
        checkpoint = torch.load(Config.MODEL_PATH, map_location=Config.DEVICE)
        state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
        model.load_state_dict(state_dict)
    except Exception as e:
        print(f"Error loading weights: {e}")
        return

    model.to(Config.DEVICE)
    model.eval()
    
    transform = A.Compose([A.Resize(Config.IMG_SIZE, Config.IMG_SIZE), ToTensorV2()])
    
    samples = find_diverse_samples(Config.DATA_ROOT, num_cases=5)
    
    for i, sample in enumerate(samples):
        img_path = sample['path']
        sample_id = sample['id']
        rows = sample['df_rows']
        
        print(f"[{i+1}/{len(samples)}] Processing: {sample_id}")
        
        img_raw = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        if img_raw is None: continue
        h_orig, w_orig = img_raw.shape[:2]
        
        img_display = enhance_image_natural(img_raw)
        img_display = np.stack([img_display]*3, axis=-1)
        
        img_input = img_raw.astype(np.float32)
        min_val, max_val = img_input.min(), img_input.max()
        if max_val > min_val: img_input = (img_input - min_val) / (max_val - min_val)
        else: img_input = img_input / 65535.0
        
        if Config.IN_CHANNELS == 3:
            img_input = np.stack([img_input, img_input, img_input], axis=-1) # (H, W, 3)
        else:
            img_input = np.expand_dims(img_input, axis=-1) # (H, W, 1)
        
        augmented = transform(image=img_input)
        tensor_input = augmented['image'].unsqueeze(0).to(Config.DEVICE)
        
        with torch.no_grad():
            logits = model(tensor_input)
            if logits.shape[-2:] != (h_orig, w_orig):
                logits = torch.nn.functional.interpolate(logits, size=(h_orig, w_orig), mode='bilinear', align_corners=False)
            
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).float().cpu().numpy()[0]
            
        gt_mask = np.zeros((3, h_orig, w_orig))
        classes_in_csv = ['large_bowel', 'small_bowel', 'stomach']
        for idx, cls in enumerate(classes_in_csv):
            rle = rows[rows['class'] == cls]['segmentation'].values
            if len(rle) > 0:
                gt_mask[idx] = rle_decode(rle[0], (h_orig, w_orig))
       
        dices = [compute_dice(preds[ch], gt_mask[ch]) for ch in range(3)]
        mean_dice = np.mean(dices)
        
        fig, axes = plt.subplots(1, 3, figsize=(20, 9)) 
        
        axes[0].imshow(img_display)
        axes[0].set_title(f"Case ID: {sample_id}\nOriginal MRI (Enhanced)", fontsize=16, pad=10)
        axes[0].axis('off')
        
        plot_mask_overlay(axes[1], img_display, gt_mask, "Ground Truth (Doctor's Label)\nRed:LB | Green:SB | Blue:Stomach")
        
        score_color = 'darkgreen' if mean_dice > 0.85 else ('orange' if mean_dice > 0.6 else 'red')
        title_str = f"Prediction (Model: {Config.ENCODER})\nDice Score: {mean_dice:.4f}\n"
        details = []
        labels = ['LB', 'SB', 'St']
        for idx, d in enumerate(dices):
            is_empty_correct = (np.sum(gt_mask[idx]) == 0 and np.sum(preds[idx]) == 0)
            mark = "*" if is_empty_correct else ""
            details.append(f"{labels[idx]}:{d:.2f}{mark}")
        title_str += f"({'  '.join(details)})"
        
        plot_mask_overlay(axes[2], img_display, preds, title_str)
        axes[2].set_title(title_str, fontsize=16, color=score_color, fontweight='bold', pad=10)

        plt.suptitle(f"Segmentation Result - {sample_id}", fontsize=20, y=0.98)
        plt.tight_layout()
        plt.subplots_adjust(top=0.88)
        
        save_path = os.path.join(Config.SAVE_DIR, f"{sample_id}_vis.png")
        plt.savefig(save_path, dpi=150)
        plt.close()
        print(f"Saved: {save_path}")

    print("Done.")

if __name__ == "__main__":
    main()