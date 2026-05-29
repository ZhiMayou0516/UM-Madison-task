# UW-Madison GI Tract Image Segmentation

本项目基于 UW-Madison GI Tract Image Segmentation 数据集，完成胃肠道 MRI 切片中的多器官语义分割任务。模型需要对单张 MRI 图像中的三类器官进行像素级预测：

- `large_bowel`：大肠
- `small_bowel`：小肠
- `stomach`：胃

项目主要用于医学图像分割课程实验与模型对比分析，代码覆盖 U-Net、ViT、TransUNet、DeepLabV3+ 和 LeViT-UNet 等多种分割模型。


## 1. Project Highlights

本项目围绕同一胃肠道 MRI 分割任务，对不同模型架构进行了实现和比较：

- 基于 U-Net 的医学图像分割 baseline；
- 基于纯 ViT 的 Transformer 分割实验；
- 基于 ResNet + ViT + CUP Decoder 的 TransUNet 混合模型；
- 基于 DeepLabV3+ 的纯 CNN 分割模型；
- 基于 LeViT-UNet 的轻量化 Transformer-CNN 混合模型；
- 支持 2D Dice、Positive Dice、3D Dice 和 HD95 等多层次评价指标；
- 使用病例级或 case-day 级划分，尽量避免相邻切片泄露到验证集。

---

## 2. Repository Structure

```text
UM-Madison-task-main/
├── README.md
├── UNETcode_final.ipynb
├── TransUnet/
│   ├── trU-2.5-script.py
│   ├── networks/
│   │   ├── vit_seg_configs.py
│   │   ├── vit_seg_modeling.py
│   │   └── vit_seg_modeling_resnet_skip.py
│   ├── pretrain/
│   │   └── placeholder.txt
│   ├── checkpoints_transunet_2_5D/
│   │   └── placeholder.txt
│   └── uw-madison-gi-tract-image-segmentation/
│       ├── train.csv
│       ├── sample_submission.csv
│       └── train/
│           └── placeholder.txt
└── levit/
    ├── deeplabv3plus.py
    ├── train_levit_unet.py
    ├── evaluate.py
    ├── evaluate_3D.py
    ├── visualize.py
    ├── LeViT_UNet_128s.py
    ├── LeViT_UNet_192.py
    └── LeViT_UNet_384.py
```

各文件作用如下：

| 文件 / 目录 | 说明 |
|---|---|
| `UNETcode_final.ipynb` | U-Net / ResNet34 Encoder U-Net 实验代码 |
| `TransUnet/trU-2.5-script.py` | ViT 与 TransUNet 的 2.5D 输入训练、验证、3D 评价与可视化脚本 |
| `TransUnet/networks/` | TransUNet / ViT 网络结构定义 |
| `TransUnet/pretrain/` | ViT 预训练权重放置目录 |
| `TransUnet/checkpoints_transunet_2_5D/` | TransUNet 训练权重与日志输出目录 |
| `levit/deeplabv3plus.py` | DeepLabV3+ 训练脚本 |
| `levit/train_levit_unet.py` | LeViT-UNet 训练脚本 |
| `levit/evaluate.py` | 2D Dice / Positive Dice 评价脚本 |
| `levit/evaluate_3D.py` | 3D Dice / HD95 评价脚本 |
| `levit/visualize.py` | 分割结果可视化脚本 |
| `levit/LeViT_UNet_*.py` | LeViT-UNet 不同规模模型定义 |

---

## 3. Dataset

本项目使用 Kaggle 的 UW-Madison GI Tract Image Segmentation 数据集。数据集中每张 MRI 切片对应一个图像文件，标注以 RLE 形式存储在 `train.csv` 中。

典型数据结构如下：

```text
uw-madison-gi-tract-image-segmentation/
├── train.csv
├── sample_submission.csv
└── train/
    ├── case2/
    │   └── case2_day1/
    │       └── scans/
    │           ├── slice_0001_266_266_1.50_1.50.png
    │           └── ...
    └── ...
```

`train.csv` 中的三类标注包括：

```text
large_bowel
small_bowel
stomach
```

代码会将 RLE 标注解码为三通道二值 mask，输出形状通常为：

```text
(C, H, W) = (3, 256, 256)
```

其中三个通道分别对应 large bowel、small bowel 和 stomach。背景不单独作为一个类别，而是由三个前景通道同时为 0 的区域隐式表示。

---

## 4. Environment

推荐使用 Python 3.9 或相近版本。可通过 conda 创建独立环境：

```bash
conda create -n uwgi python=3.9 -y
conda activate uwgi
```

安装主要依赖：

```bash
pip install numpy pandas matplotlib tqdm scikit-learn scipy
pip install opencv-python pillow albumentations
pip install torch torchvision
pip install segmentation-models-pytorch timm ml-collections medpy ptflops
```

如果使用 CUDA，请根据本机显卡和 CUDA 版本安装匹配的 PyTorch 版本。

---

## 5. Usage

### 5.1 U-Net

U-Net 实验代码位于：

```text
UNETcode_final.ipynb
```

运行前需要确认 notebook 中的数据路径：

```python
DATA_ROOT = "./"
TRAIN_CSV = os.path.join(DATA_ROOT, "train.csv")
```

如果数据不在当前目录，需要将 `DATA_ROOT` 修改为实际数据集路径。

---

### 5.2 ViT / TransUNet

TransUNet 主脚本位于：

```text
TransUnet/trU-2.5-script.py
```

运行前先修改脚本中的路径配置：

```python
class CFG:
    data_root = r"your/path/to/uw-madison-gi-tract-image-segmentation"
    train_csv = os.path.join(data_root, "train.csv")
    train_dir = os.path.join(data_root, "train")
    out_dir = "./checkpoints_transunet_2_5D"
```

然后进入 `TransUnet/` 目录运行：

```bash
cd TransUnet
python trU-2.5-script.py
```

默认配置包括：

```text
input mode: 2.5D, using slices [k-1, k, k+1]
image size: 256 × 256
num classes: 3
batch size: 4
epochs: 15
optimizer: Adam
loss: 0.5 × Dice Loss + 0.5 × BCE
scheduler: StepLR, step_size=5, gamma=0.5
```

如果没有 ViT 的 `.npz` 预训练权重，可以先将：

```python
use_pretrained = False
```

或者将对应权重文件放入：

```text
TransUnet/pretrain/
```

---

### 5.3 DeepLabV3+

DeepLabV3+ 训练脚本位于：

```text
levit/deeplabv3plus.py
```

运行前修改 `Config` 中的数据路径、权重路径和 backbone：

```python
class Config:
    DATA_ROOT = "/path/to/uw-madison-gi-tract-image-segmentation/"
    encoder = "mobilenet_v2"   # or resnet50 / efficientnet-b4
    batch_size = 16
    epochs = 50
    lr = 1e-4
```

运行：

```bash
cd levit
python deeplabv3plus.py
```

可尝试的 encoder 包括：

```text
resnet50
efficientnet-b4
mobilenet_v2
```

---

### 5.4 LeViT-UNet

LeViT-UNet 训练脚本位于：

```text
levit/train_levit_unet.py
```

在 `Config` 中选择模型规模：

```python
MODEL_NAME = "LeViT_UNet_384"
```

可选模型包括：

```text
LeViT_UNet_128s
LeViT_UNet_192
LeViT_UNet_384
```

运行：

```bash
cd levit
python train_levit_unet.py
```

注意：`LeViT_UNet_*.py` 中存在 `import utils`。如果本地环境中没有对应的 `utils.py`，需要补齐原始 LeViT 代码中的工具文件，或在确认 `fuse=False` 的前提下移除相关 batchnorm fuse 逻辑。

---

### 5.5 Evaluation

2D 评价：

```bash
cd levit
python evaluate.py
```

3D 评价：

```bash
cd levit
python evaluate_3D.py
```

可视化预测结果：

```bash
cd levit
python visualize.py
```

运行评价脚本前，需要修改对应脚本中的模型路径：

```python
MODEL_PATH = "weights/best_model_xxx.pth"
```

---

## 6. Evaluation Metrics

本项目采用多层次指标评价模型性能：

| 指标 | 说明 |
|---|---|
| Dice All | 在所有切片上计算 Dice，包括空切片 |
| Positive Dice | 仅在存在真实器官标注的切片上计算 Dice，更能反映有效分割能力 |
| 3D Dice | 将 2D 预测按病例重建为 3D 体数据后计算体素级 Dice |
| HD95 | 计算 95% Hausdorff Distance，用于评价 3D 边界误差 |

由于医学图像分割任务中空切片比例较高，仅使用 Dice All 容易造成指标虚高。因此，本项目同时报告 Positive Dice、3D Dice 和 HD95，以更完整地评价模型在有效器官区域和三维空间连续性上的表现。

---

## 7. Experimental Results

### 7.1 U-Net Baseline

U-Net 实验经过多轮修正后，主要改进包括：

- 修正 RLE mask 解码和转置错误；
- 正确读取 16-bit MRI 图像；
- 使用 RGB 叠加方式进行分割可视化；
- 改进 loss 与 dice 的计算方式；
- 尝试 ResNet34 预训练 encoder 与冻结-解冻训练策略。

最终 U-Net baseline 在 50 epoch 设置下取得了约 `0.77` 的 Dice 结果。该结果作为后续 Transformer 和混合模型实验的基础参照。

### 7.2 ViT and TransUNet

| Model | Dice 2D All | Dice 2D Positive | Dice 3D | HD95 |
|---|---:|---:|---:|---:|
| ViT-B/16 | 0.8896 | 0.6825 | 0.7761 | 7.3178 |
| ViT-B/32 | 0.8490 | 0.5663 | 0.6656 | 9.5140 |
| R50-ViT-B/16 | 0.9154 | 0.7514 | 0.8248 | 6.6725 |
| R50-ViT-L/32 | 0.9019 | 0.7185 | 0.8104 | 7.2686 |
| R26-ViT-B/32 | 0.9066 | 0.7275 | 0.8121 | 6.8804 |

实验结果显示，ViT-B/16 相比 ViT-B/32 具有更好的空间细节表达能力。TransUNet 在纯 ViT 基础上引入 ResNet 局部特征和 U 型解码器后，整体分割性能进一步提升。其中 `R50-ViT-B/16` 在本组实验中表现最好。

### 7.3 DeepLabV3+ and LeViT-UNet

| Architecture | Backbone | Dice All | Dice Positive | 3D Dice | HD95 |
|---|---|---:|---:|---:|---:|
| DeepLabV3+ | ResNet50 | 0.8730 | 0.6734 | 0.7726 | 8.659 |
| DeepLabV3+ | EfficientNet-B4 | 0.8746 | 0.7487 | 0.7713 | 9.560 |
| DeepLabV3+ | MobileNetV2 | 0.8778 | 0.6913 | 0.7445 | 9.510 |
| LeViT-UNet | 128s | 0.8458 | 0.6292 | 0.6830 | 11.897 |
| LeViT-UNet | 192 | 0.8510 | 0.6756 | 0.7167 | 12.572 |
| LeViT-UNet | 384 | 0.8637 | 0.6969 | 0.7379 | 10.235 |

在当前实验设置下，DeepLabV3+ 系列整体优于从零训练的 LeViT-UNet 系列，说明小样本医学图像分割中，带有预训练权重的 CNN backbone 仍然具有较强稳定性。LeViT-UNet 随模型规模增大表现逐步提升，说明轻量化 Transformer-CNN 结构仍具有进一步优化空间。

---



5. **结果复现依赖数据划分和路径设置**  
   不同脚本中数据划分方式略有差异，包括 case-day 级划分和 GroupKFold 病例级划分。对比实验时应保证数据划分、输入尺寸、训练轮数和评价协议一致。

---

