import argparse
import os
import cv2
import json
from PIL import Image
import base64
import zlib
from io import BytesIO
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score, roc_curve
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.ndimage import gaussian_filter, map_coordinates

def _check_mask(arr, name):
    arr = np.asarray(arr)
    if arr.ndim not in (2, 3):
        raise ValueError(f"{name} must be 2-D (H×W) or 3-D (N×H×W), got shape {arr.shape}")
    return arr
 
def compute_dice(mask_true, mask_pred):
    """Dice coefficient = F1 score for segmentation masks.
 
    Dice = 2·TP / (2·TP + FP + FN)
    Equivalent to the pixel-level F1 score.
    """
    t, p = _flatten_pair(
        _check_mask(mask_true, "mask_true"),
        _check_mask(mask_pred, "mask_pred"),
    )
    tp, tn, fp, fn = _confusion(t, p)
    denom = 2 * tp + fp + fn
    return (2 * tp) / denom if denom > 0 else 1.0   # perfect if both empty


def compute_iou(mask_true, mask_pred):
    """Intersection over Union (Jaccard index) — bonus metric often used alongside Dice."""
    t, p = _flatten_pair(
        _check_mask(mask_true, "mask_true"),
        _check_mask(mask_pred, "mask_pred"),
    )
    tp, tn, fp, fn = _confusion(t, p)
    denom = tp + fp + fn
    return tp / denom if denom > 0 else 1.0

def _confusion(mask_true_flat, mask_pred_flat):
    tp = np.sum( mask_true_flat &  mask_pred_flat)
    tn = np.sum(~mask_true_flat & ~mask_pred_flat)
    fp = np.sum(~mask_true_flat &  mask_pred_flat)
    fn = np.sum( mask_true_flat & ~mask_pred_flat)
    return tp, tn, fp, fn

def _flatten_pair(mask_true, mask_pred):
    """Return two 1-D boolean arrays regardless of input shape."""
    return mask_true.ravel().astype(bool), mask_pred.ravel().astype(bool)
 
def compute_sensitivity(mask_true, mask_pred):
    """Sensitivity = TP / (TP + FN)  — pixel-level true positive rate."""
    t, p = _flatten_pair(
        _check_mask(mask_true, "mask_true"),
        _check_mask(mask_pred, "mask_pred"),
    )
    tp, tn, fp, fn = _confusion(t, p)
    return tp / (tp + fn) if (tp + fn) > 0 else 0.0
 
def compute_specificity(mask_true, mask_pred):
    """Specificity = TN / (TN + FP)  — pixel-level true negative rate."""
    t, p = _flatten_pair(
        _check_mask(mask_true, "mask_true"),
        _check_mask(mask_pred, "mask_pred"),
    )
    tp, tn, fp, fn = _confusion(t, p)
    return tn / (tn + fp) if (tn + fp) > 0 else 0.0
 
def compute_auc_roc(mask_true, mask_scores):
    """Pixel-level AUC-ROC using soft probability scores."""
    t = _check_mask(mask_true,   "mask_true").ravel().astype(int)
    s = _check_mask(mask_scores, "mask_scores").ravel().astype(float)
    if len(np.unique(t)) < 2:
        raise ValueError("mask_true has only one class — AUC-ROC is undefined.")
    return roc_auc_score(t, s)

def segmentation_metrics(
    mask_true,
    mask_pred,
    mask_scores=None,
    threshold=0.5,
    verbose=True,
):
    """
    Compute all metrics for U-Net / binary segmentation output.
 
    Args:
        mask_true    : ground-truth binary mask  (H×W) or (N×H×W)
        mask_pred    : predicted binary mask OR soft probability map (H×W) or (N×H×W).
                       If float values are detected, thresholded at `threshold`.
        mask_scores  : soft probability map for AUC-ROC. If None and mask_pred
                       contains floats, mask_pred is used as scores automatically.
        threshold    : binarisation cutoff when mask_pred is a probability map (default 0.5)
        verbose      : print a formatted summary table
 
    Returns:
        dict with keys: sensitivity, specificity, dice (f1), iou, auc_roc
    """
    mask_true   = _check_mask(mask_true,  "mask_true")
    mask_pred   = _check_mask(mask_pred,  "mask_pred")
 
    # Auto-detect soft predictions
    is_soft = mask_pred.dtype.kind == "f" or mask_pred.max() > 1
    if is_soft:
        if mask_scores is None:
            mask_scores = mask_pred.copy()
        mask_pred = (mask_pred >= threshold).astype(np.uint8)
 
    sensitivity = compute_sensitivity(mask_true, mask_pred)
    specificity = compute_specificity(mask_true, mask_pred)
    dice        = compute_dice(mask_true, mask_pred)
    iou         = compute_iou(mask_true, mask_pred)
    auc_roc     = compute_auc_roc(mask_true, mask_scores) if mask_scores is not None else None
 
    if verbose:
        sep = "─" * 42
        print(sep)
        print(f"{'SEGMENTATION METRICS (pixel-level)':^42}")
        print(sep)
        print(f"  Sensitivity (Recall)  : {sensitivity:.4f}")
        print(f"  Specificity           : {specificity:.4f}")
        print(f"  Dice / F1             : {dice:.4f}")
        print(f"  IoU  (Jaccard)        : {iou:.4f}")
        if auc_roc is not None:
            print(f"  AUC-ROC               : {auc_roc:.4f}")
        else:
            print(f"  AUC-ROC               : N/A (no score map provided)")
        print(sep)
 
    return {
        "sensitivity": sensitivity,
        "specificity": specificity,
        "dice":        dice,
        "iou":         iou,
        "auc_roc":     auc_roc,
    }

def dataset_metrics(mask_true_list, mask_pred_list, mask_scores_list=None, threshold=0.5):
    """
    Compute per-image metrics and dataset-level mean ± std.
 
    Args:
        mask_true_list   : list of (H×W) ground-truth masks
        mask_pred_list   : list of (H×W) predicted masks or probability maps
        mask_scores_list : list of (H×W) soft score maps (optional)
        threshold        : binarisation threshold
 
    Returns:
        per_image : list of per-image metric dicts
        summary   : dict of mean / std for each metric
    """
    if mask_scores_list is None:
        mask_scores_list = [None] * len(mask_true_list)
 
    per_image = [
        segmentation_metrics(t, p, s, threshold=threshold, verbose=False)
        for t, p, s in zip(mask_true_list, mask_pred_list, mask_scores_list)
    ]
 
    keys = [k for k in per_image[0] if per_image[0][k] is not None]
    summary = {}
    for k in keys:
        vals = [m[k] for m in per_image if m[k] is not None]
        summary[k] = {"mean": float(np.mean(vals)), "std": float(np.std(vals))}
 
    sep = "─" * 50
    print(sep)
    print(f"{'DATASET-LEVEL METRICS  (n=' + str(len(per_image)) + ')':^50}")
    print(sep)
    for k, v in summary.items():
        print(f"  {k:<22}: {v['mean']:.4f}  ±  {v['std']:.4f}")
    print(sep)
 
    return per_image, summary

def iou_score(pred_mask, true_mask, num_classes=2):
    """Mean Intersection-over-Union across all classes."""
    ious = []
    for c in range(num_classes):
        pred_c = (pred_mask == c)
        true_c = (true_mask == c)
        intersection = (pred_c & true_c).sum().float()
        union = (pred_c | true_c).sum().float()
        if union == 0:
            ious.append(torch.tensor(1.0))
        else:
            ious.append(intersection / union)
    return torch.stack(ious).mean().item()

def dice_score(pred_mask, true_mask, eps=1e-6):
    pred_mask = torch.as_tensor(pred_mask)
    true_mask = torch.as_tensor(true_mask)

    pred_fg = (pred_mask == 1).float()
    true_fg = (true_mask == 1).float()

    intersection = (pred_fg * true_fg).sum()

    dice = (2.0 * intersection + eps) / (
        pred_fg.sum() + true_fg.sum() + eps
    )

    return dice.item()

def dice_loss(logits, targets, eps=1e-6): #F1 ~= Dice

    probs = torch.softmax(logits, dim=1)[:, 1]   # [B, H, W]
    fg    = (targets == 1).float()
    intersection = (probs * fg).sum()
    dice = 1.0 - (2.0 * intersection + eps) / (probs.sum() + fg.sum() + eps)

    return dice

def Loss_criterion(logits, targets):    

    # BCE Loss
    if targets.ndim == 4:
        targets = targets.squeeze(1)          # [B, 1, H, W] → [B, H, W]
    targets = targets.long() 
    
    CE = F.cross_entropy(logits, targets) 
    dice = dice_loss(logits, targets)
    l = CE + dice
    return l

def Training(model, scheduler, NUM_EPOCHS, DEVICE, deep_supervission, train_loader, val_loader, optimizer, name):
    history = {'train_loss': [], 'val_loss': [], 'val_iou': [], 'val_dice': []}

    print(f'Training U-Net for {NUM_EPOCHS} epochs on {DEVICE}...')
    print(f'{"Epoch":>6}  {"Train Loss":>12}  {"Val Loss":>10}  {"Val IoU":>10}  {"Val Dice":>10}')
    print('-' * 60)

    for epoch in range(1, NUM_EPOCHS + 1):
        train_loss = train_one_epoch(model, deep_supervission, train_loader, optimizer, DEVICE)
        val_loss, val_iou, val_dice = evaluate(model,deep_supervission, val_loader, DEVICE)
        
        scheduler.step(val_loss)

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_iou'].append(val_iou)
        history['val_dice'].append(val_dice)

        if epoch % 2 == 0 or epoch == 1:
            print(f'{epoch:>6}  {train_loss:>12.4f}  {val_loss:>10.4f}  {val_iou:>10.4f}  {val_dice:>10.4f}')
    print('\nTraining complete!')
    torch.save(model.state_dict(), 'model_weights.pth')
    with open(f"history_{name}.json", "w") as f:
        json.dump(history, f)
    return model, history

def train_one_epoch(model,deep_supervission, loader, optimizer, device):
    model.train()
    total_loss = 0.0
    for images, masks in loader:
        images  = images.to(device)
        masks   = masks.to(device)

        optimizer.zero_grad()
        logits = model(images)
        if deep_supervission:
            loss = 0
            for out in logits:
                loss += Loss_criterion(out, masks)
            loss /= len(logits)
        else:        
            loss = Loss_criterion(logits, masks)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

@torch.no_grad()
def evaluate(model,deep_supervission, loader, device):
    model.eval()
    total_loss, total_iou, total_dice = 0.0, 0.0, 0.0
    for images, masks in loader:
        images  = images.to(device)
        masks   = masks.to(device)
  

        logits = model(images)

        if deep_supervission:
            loss = 0
            for out in logits:
                loss += Loss_criterion(out, masks)
            loss /= len(logits)
        else:
            loss = Loss_criterion(logits, masks)

        if deep_supervission:
            preds = logits[-1].argmax(dim=1)

        else :
            preds = logits.argmax(dim=1)

        total_loss += loss.item()

        total_iou  += iou_score(preds, masks)
        total_dice += dice_score(preds, masks)

    n = len(loader)
    return total_loss / n, total_iou / n, total_dice / n

class DoubleConv(nn.Module): 
    """Two (3x3 Conv -> BN -> ReLU) blocks — the basic U-Net building block."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)

class EncoderBlock(nn.Module):
    """DoubleConv + MaxPool. Returns both the skip-connection feature map and the pooled output."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = DoubleConv(in_channels, out_channels)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        skip = self.conv(x)       # high-res feature map (skip connection)
        pooled = self.pool(skip)  # downsampled output
        return skip, pooled

class DecoderBlock(nn.Module):
    """Upsample (transposed conv) -> concat skip -> DoubleConv."""

    def __init__(self, in_channels: int, up_channels_in: int,up_channels_out: int, out_channels: int):
        super().__init__()
        self.up = nn.ConvTranspose2d(up_channels_in, up_channels_out, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)  # in_channels because of concat

    def forward(self, x, skip):
        # print(x.shape)

        x = self.up(x)

        # print(x.shape)
        # print(skip.shape)

        # Align sizes in case of odd dimensions
        if x.shape != skip.shape:
            x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)
        x = torch.cat([skip, x], dim=1)  # channel-wise concatenation
        # print(x.shape)
        return self.conv(x)


class UNet(nn.Module):
    """
    U-Net for binary or multi-class segmentation.

    Parameters
    ----------
    in_channels  : number of input image channels (1 for grayscale, 3 for RGB)
    out_channels : number of segmentation classes (2 for binary)
    base_features: feature channels at the first encoder level (doubles each level)
    """

    def __init__(self, in_channels: int = 3, out_channels: int = 2, base_features: int = 64):
        super().__init__()
        f = base_features

        # ── Encoder ──────────────────────────────────────────────────────────
        self.enc1 = EncoderBlock(in_channels, f)       # 64
        self.enc2 = EncoderBlock(f, f * 2)             # 128
        self.enc3 = EncoderBlock(f * 2, f * 4)         # 256
        self.enc4 = EncoderBlock(f * 4, f * 8)         # 512

        # ── Bottleneck ───────────────────────────────────────────────────────
        self.bottleneck = DoubleConv(f * 8, f * 16)    # 1024

        # ── Decoder ──────────────────────────────────────────────────────────
        self.dec4 = DecoderBlock(f * 16,f * 16,f * 8, f * 8)        # 512
        self.dec3 = DecoderBlock(f * 8,f * 8,f * 4, f * 4)         # 256
        self.dec2 = DecoderBlock(f * 4,f * 4,f * 2, f * 2)         # 128
        self.dec1 = DecoderBlock(f * 2,f * 2,f, f)             # 64

        # ── Output ───────────────────────────────────────────────────────────
        self.out_conv = nn.Conv2d(f, out_channels, kernel_size=1)

    def forward(self, x):
        # Encoder
        s1, x = self.enc1(x)
        s2, x = self.enc2(x)
        s3, x = self.enc3(x)
        s4, x = self.enc4(x)

        # Bottleneck
        x = self.bottleneck(x)

        # Decoder (skip connections passed in)
        x = self.dec4(x, s4)
        x = self.dec3(x, s3)
        x = self.dec2(x, s2)
        x = self.dec1(x, s1)

        return self.out_conv(x)  # raw logits (H x W x out_channels)

class UNetpp_4(nn.Module):
    def __init__(self, 
                 in_channels: int = 1, 
                 out_channels: int = 2, 
                 base_features: int = 64,
                 deep_supervision: bool = False,
                 ):
        super().__init__()
        
        #Features        
        f = base_features
        f_list = [f, f*2, f*4, f*8, f*16] #[64,128,256,512,1024]
        #Deep Supervision
        self.deep_supervision = deep_supervision

        # ── Encoder ──────────────────────────────────────────────────────────
        self.enc0_0 = EncoderBlock(in_channels, f_list[0])       # 64
        self.enc1_0 = EncoderBlock(f_list[0]  , f_list[1])       # 128
        self.enc2_0 = EncoderBlock(f_list[1]  , f_list[2])       # 256
        self.enc3_0 = EncoderBlock(f_list[2]  , f_list[3])       # 512
        self.enc4_0 = DoubleConv(f_list[3]    , f_list[4])       # 1024

        # Nesting
        self.dec0_1 = DecoderBlock(f_list[0]   + f_list[1], f_list[1], f_list[1], f_list[0])
        self.dec0_2 = DecoderBlock(f_list[0]*2 + f_list[1], f_list[1], f_list[1], f_list[0])
        self.dec0_3 = DecoderBlock(f_list[0]*3 + f_list[1], f_list[1], f_list[1], f_list[0])
        self.dec1_1 = DecoderBlock(f_list[1]   + f_list[2], f_list[2], f_list[2], f_list[1])
        self.dec1_2 = DecoderBlock(f_list[1]*2 + f_list[2], f_list[2], f_list[2], f_list[1])
        self.dec2_1 = DecoderBlock(f_list[2]   + f_list[3], f_list[3], f_list[3], f_list[2])
        self.dec3_1 = DecoderBlock(f_list[3]   + f_list[4],f_list[4], f_list[4], f_list[3])        # 512
        self.dec2_2 = DecoderBlock(f_list[2]*2 + f_list[3],f_list[3], f_list[3], f_list[2])
        self.dec1_3 = DecoderBlock(f_list[1]*3 + f_list[2],f_list[2], f_list[2], f_list[1])         # 128
        self.dec0_4 = DecoderBlock(f_list[0]*4 + f_list[1],f_list[1], f_list[1], f_list[0])             # 64

        # ── Output ───────────────────────────────────────────────────────────
        if self.deep_supervision:
            self.out_conv_1 = nn.Conv2d(f_list[0], out_channels, kernel_size=1)
            self.out_conv_2 = nn.Conv2d(f_list[0], out_channels, kernel_size=1)
            self.out_conv_3 = nn.Conv2d(f_list[0], out_channels, kernel_size=1)
            self.out_conv_4 = nn.Conv2d(f_list[0], out_channels, kernel_size=1)
        else:
            self.out_conv = nn.Conv2d(f_list[0], out_channels, kernel_size=1)

    def forward(self, x):
        #Encoder - outputs
        #(Conv, downsample)

        # Encoders - "Backbone"
        x_0_0, xd_0_0 = self.enc0_0(x)
        x_1_0, xd_1_0 = self.enc1_0(xd_0_0)
        x_2_0, xd_2_0 = self.enc2_0(xd_1_0)
        x_3_0, xd_3_0 = self.enc3_0(xd_2_0)
        x_4_0 = self.enc4_0(xd_3_0)

        #Decoder - inputs
        #(Upsample, concat)

        #Nesting (Upsample, concat)
        x_0_1 = self.dec0_1(x_1_0, x_0_0)

        x_1_1 = self.dec1_1(x_2_0, x_1_0)
        x_2_1 = self.dec2_1(x_3_0, x_2_0)
        x_3_1 = self.dec3_1(x_4_0, x_3_0)
        x_0_2 = self.dec0_2(x_1_1, torch.cat([x_0_0, x_0_1],dim=1))
        x_1_2 = self.dec1_2(x_2_1, torch.cat([x_1_0, x_1_1],dim=1))
        x_2_2 = self.dec2_2(x_3_1, torch.cat([x_2_0, x_2_1],dim=1))
        x_0_3 = self.dec0_3(x_1_2, torch.cat([x_0_0,x_0_1,x_0_2],dim=1))
        x_1_3 = self.dec1_3(x_2_2, torch.cat([x_1_0,x_1_1,x_1_2],dim=1))
        x_0_4 = self.dec0_4(x_1_3, torch.cat([x_0_0,x_0_1,x_0_2,x_0_3],dim=1))

        if self.deep_supervision:
            Out_1 = self.out_conv_1(x_0_1)
            Out_2 = self.out_conv_2(x_0_2)
            Out_3 = self.out_conv_3(x_0_3)
            Out_4 = self.out_conv_4(x_0_4)
            return [Out_1,Out_2,Out_3,Out_4]
        else:   
            return self.out_conv(x_0_4)  # raw logits (H x W x out_channels)

class UNetpp_3(nn.Module):
    def __init__(self, 
                 in_channels: int = 1, 
                 out_channels: int = 2, 
                 base_features: int = 64,
                 deep_supervision: bool = False,
                 ):
        super().__init__()
        
        #Features        
        f = base_features
        f_list = [f, f*2, f*4, f*8] #[64,128,256,512,1024]
        #Deep Supervision
        self.deep_supervision = deep_supervision

        # ── Encoder ──────────────────────────────────────────────────────────
        self.enc0_0 = EncoderBlock(in_channels, f_list[0])       # 64
        self.enc1_0 = EncoderBlock(f_list[0]  , f_list[1])       # 128
        self.enc2_0 = EncoderBlock(f_list[1]  , f_list[2])       # 256
        self.enc3_0 = DoubleConv(f_list[2]  , f_list[3])       # 512


        # Nesting
        self.dec0_1 = DecoderBlock(f_list[0]   + f_list[1], f_list[1], f_list[1], f_list[0])
        self.dec0_2 = DecoderBlock(f_list[0]*2 + f_list[1], f_list[1], f_list[1], f_list[0])
        self.dec0_3 = DecoderBlock(f_list[0]*3 + f_list[1], f_list[1], f_list[1], f_list[0])
        self.dec1_1 = DecoderBlock(f_list[1]   + f_list[2], f_list[2], f_list[2], f_list[1])
        self.dec1_2 = DecoderBlock(f_list[1]*2 + f_list[2], f_list[2], f_list[2], f_list[1])
        self.dec2_1 = DecoderBlock(f_list[2]   + f_list[3], f_list[3], f_list[3], f_list[2])


        # ── Output ───────────────────────────────────────────────────────────
        if self.deep_supervision:
            self.out_conv_1 = nn.Conv2d(f_list[0], out_channels, kernel_size=1)
            self.out_conv_2 = nn.Conv2d(f_list[0], out_channels, kernel_size=1)
            self.out_conv_3 = nn.Conv2d(f_list[0], out_channels, kernel_size=1)
        else:
            self.out_conv = nn.Conv2d(f_list[0], out_channels, kernel_size=1)

    def forward(self, x):
        #Encoder - outputs
        #(Conv, downsample)

        # Encoders - "Backbone"
        x_0_0, xd_0_0 = self.enc0_0(x)
        x_1_0, xd_1_0 = self.enc1_0(xd_0_0)
        x_2_0, xd_2_0 = self.enc2_0(xd_1_0)
        x_3_0 = self.enc3_0(xd_2_0)


        #Decoder - inputs
        #(Upsample, concat)

        #Nesting (Upsample, concat)
        x_0_1 = self.dec0_1(x_1_0, x_0_0)

        x_1_1 = self.dec1_1(x_2_0, x_1_0)
        x_2_1 = self.dec2_1(x_3_0, x_2_0)
        x_0_2 = self.dec0_2(x_1_1, torch.cat([x_0_0, x_0_1],dim=1))
        x_1_2 = self.dec1_2(x_2_1, torch.cat([x_1_0, x_1_1],dim=1))
        x_0_3 = self.dec0_3(x_1_2, torch.cat([x_0_0,x_0_1,x_0_2],dim=1))

        if self.deep_supervision:
            Out_1 = self.out_conv_1(x_0_1)
            Out_2 = self.out_conv_2(x_0_2)
            Out_3 = self.out_conv_3(x_0_3)
            return [Out_1,Out_2,Out_3]
        else:   
            return self.out_conv(x_0_3)  # raw logits (H x W x out_channels)

class UNetpp_2(nn.Module):
    def __init__(self, 
                 in_channels: int = 1, 
                 out_channels: int = 2, 
                 base_features: int = 64,
                 deep_supervision: bool = False,
                 ):
        super().__init__()
        
        #Features        
        f = base_features
        f_list = [f, f*2, f*4] #[64,128,256,512,1024]
        #Deep Supervision
        self.deep_supervision = deep_supervision

        # ── Encoder ──────────────────────────────────────────────────────────
        self.enc0_0 = EncoderBlock(in_channels, f_list[0])       # 64
        self.enc1_0 = EncoderBlock(f_list[0]  , f_list[1])       # 128
        self.enc2_0 = DoubleConv(f_list[1]  , f_list[2])       # 256



        # Nesting
        self.dec0_1 = DecoderBlock(f_list[0]   + f_list[1], f_list[1], f_list[1], f_list[0])
        self.dec0_2 = DecoderBlock(f_list[0]*2 + f_list[1], f_list[1], f_list[1], f_list[0])
        self.dec1_1 = DecoderBlock(f_list[1]   + f_list[2], f_list[2], f_list[2], f_list[1])



        # ── Output ───────────────────────────────────────────────────────────
        if self.deep_supervision:
            self.out_conv_1 = nn.Conv2d(f_list[0], out_channels, kernel_size=1)
            self.out_conv_2 = nn.Conv2d(f_list[0], out_channels, kernel_size=1)
        else:
            self.out_conv = nn.Conv2d(f_list[0], out_channels, kernel_size=1)

    def forward(self, x):
        #Encoder - outputs
        #(Conv, downsample)

        # Encoders - "Backbone"
        x_0_0, xd_0_0 = self.enc0_0(x)
        x_1_0, xd_1_0 = self.enc1_0(xd_0_0)
        x_2_0 = self.enc2_0(xd_1_0)



        #Decoder - inputs
        #(Upsample, concat)

        #Nesting (Upsample, concat)
        x_0_1 = self.dec0_1(x_1_0, x_0_0)
        x_1_1 = self.dec1_1(x_2_0, x_1_0)
        x_0_2 = self.dec0_2(x_1_1, torch.cat([x_0_0, x_0_1],dim=1))



        if self.deep_supervision:
            Out_1 = self.out_conv_1(x_0_1)
            Out_2 = self.out_conv_2(x_0_2)
            return [Out_1,Out_2]
        else:   
            return self.out_conv(x_0_2)  # raw logits (H x W x out_channels)

class UNetpp_1(nn.Module):
    def __init__(self, 
                 in_channels: int = 1, 
                 out_channels: int = 2, 
                 base_features: int = 64,
                 deep_supervision: bool = False,
                 ):
        super().__init__()
        
        #Features        
        f = base_features
        f_list = [f, f*2] #[64,128]
        #Deep Supervision
        self.deep_supervision = deep_supervision

        # ── Encoder ──────────────────────────────────────────────────────────
        self.enc0_0 = EncoderBlock(in_channels, f_list[0])       # 64
        self.enc1_0 = DoubleConv(f_list[0]  , f_list[1])       # 128

        # Nesting
        self.dec0_1 = DecoderBlock(f_list[0]   + f_list[1], f_list[1], f_list[1], f_list[0])


        # ── Output ───────────────────────────────────────────────────────────
        if self.deep_supervision:
            self.out_conv_1 = nn.Conv2d(f_list[0], out_channels, kernel_size=1)
        else:
            self.out_conv = nn.Conv2d(f_list[0], out_channels, kernel_size=1)

    def forward(self, x):
        #Encoder - outputs
        #(Conv, downsample)

        # Encoders - "Backbone"
        x_0_0, xd_0_0 = self.enc0_0(x)
        x_1_0 = self.enc1_0(xd_0_0)
        



        #Decoder - inputs
        #(Upsample, concat)

        #Nesting (Upsample, concat)
        x_0_1 = self.dec0_1(x_1_0, x_0_0)
    

        if self.deep_supervision:
            Out_1 = self.out_conv_1(x_0_1)
            return [Out_1]
        else:   
            return self.out_conv(x_0_1)  # raw logits (H x W x out_channels)

class veinDataset(Dataset):
    def __init__(self, 
                 dir: str = '.',
                 training: bool = True,
                 augment: bool = True,
                 Drive: int = 1,
                 Actor: int =1,
                 augment_amount: int = 2,
                 ENHANCE: bool = True):
        if Drive == 1:
            dir_ext = dir + 'STARE/images/'
            self.Imagenes_nombres = sorted(os.listdir(dir_ext))
            self.data = [np.array(Image.open(dir_ext+fname)) for fname in self.Imagenes_nombres]
            if Actor == 1:
                dir_ext = dir + 'STARE/masks_1/'
                self.Mascaras_nombres = sorted(os.listdir(dir_ext))
                self.masks = [np.array(Image.open(dir_ext+fname)) for fname in self.Mascaras_nombres]
            else:
                dir_ext = dir + 'STARE/masks_2/'
                self.Mascaras_nombres = sorted(os.listdir(dir_ext))
                self.masks = [np.array(Image.open(dir_ext+fname)) for fname in self.Mascaras_nombres]                
            self.filters = []

            if ENHANCE:
                buffer1 = []
                for image in self.data:
                    buffer1.append(cv2.convertScaleAbs(image, alpha=1, beta=50))
                self.data = buffer1

                for image in self.data:
                    self.filters.append(self._auto_mask_filter(image.copy(),100))
            else:
                for image in self.data:
                    self.filters.append(self._auto_mask_filter(image.copy(),40))

            for i in range(len(self.data)):
                self.data[i] = cv2.bitwise_and(self.data[i], self.data[i], mask=self.filters[i])

            buffer1  = [cv2.resize(img, (512,512), interpolation=cv2.INTER_AREA) for img in self.data]
            buffer2  = [cv2.resize(img.astype('uint8'), (512,512), interpolation=cv2.INTER_AREA) for img in self.masks]        

            self.data = buffer1
            self.masks = buffer2

            self.n_samples = len(self.data)

        elif Drive == 2:
            dir_ext = dir + 'CHASE_DB1/images/'
            self.Imagenes_nombres = sorted(os.listdir(dir_ext))
            self.data = [np.array(Image.open(dir_ext+fname)) for fname in self.Imagenes_nombres]
            dir_ext = dir + 'CHASE_DB1/mask/'
            self.Mascaras_nombres = sorted(os.listdir(dir_ext))
            self.masks = [self._extract_bit_map(dir_ext+fname) for fname in self.Mascaras_nombres]
            self.filters = []


            if ENHANCE:
                buffer1 = []
                for image in self.data:
                    buffer1.append(cv2.convertScaleAbs(image, alpha=1.2, beta=100))
                self.data = buffer1

                for image in self.data:
                    self.filters.append(self._auto_mask_filter(image.copy(),100))
            else:
                for image in self.data:
                    self.filters.append(self._auto_mask_filter(image.copy(),5))

            for i in range(len(self.data)):
                self.data[i] = cv2.bitwise_and(self.data[i], self.data[i], mask=self.filters[i])

            buffer1  = [cv2.resize(img, (512,512), interpolation=cv2.INTER_AREA) for img in self.data]
            buffer2  = [cv2.resize(img.astype('uint8'), (512,512), interpolation=cv2.INTER_AREA) for img in self.masks]
            self.data = buffer1
            self.masks = buffer2

            self.n_samples = len(self.data)
        else:
            dir_ext = dir + 'DRIVE/train/images/'
            self.Imagenes_nombres = sorted(os.listdir(dir_ext))
            self.data = [np.array(Image.open(dir_ext+fname)) for fname in self.Imagenes_nombres]
            
            dir_ext = dir + 'DRIVE/train/mask/'
            self.Mascaras_nombres = sorted(os.listdir(dir_ext))
            self.masks = [np.array(Image.open(dir_ext+fname)) for fname in self.Mascaras_nombres]
            
            dir_ext = dir + 'DRIVE/train/filter/'
            self.filtros_nombres = sorted(os.listdir(dir_ext))
            self.filters = [np.array(Image.open(dir_ext+fname)) for fname in self.filtros_nombres]

            for i in range(len(self.data)):
                self.data[i] = cv2.bitwise_and(self.data[i], self.data[i], mask=self.filters[i])

            # self.shuffle = np.random.permutation(len(self.data))

            # self.data = self.data[self.shuffle]
            # self.masks = self.masks[self.shuffle]
            len_data  = len(self.data)

            if training:    
                self.data = self.data[0:int(len_data*0.7)]
                self.masks = self.masks[0:int(len_data*0.7)]
                
            else:
                self.data = self.data[int(len_data*0.7):len_data]
                self.masks = self.masks[int(len_data*0.7):len_data]  

            augmented_data = []
            augmented_masks = []

            if augment:
                for i in range(len(self.data)):
                    image = self.data[i]
                    mask = self.masks[i]

                    augmented_data.append(image)
                    augmented_masks.append(mask)
                    for j in range(augment_amount):
                        image, mask = self._augmentation(image.copy(), mask.copy())

                        image, mask = self._elastic_transform(image, mask)
                        augmented_data.append(image)
                        augmented_masks.append(mask)

                self.data = augmented_data
                self.masks = augmented_masks

            buffer1  = [cv2.resize(img, (512,512), interpolation=cv2.INTER_AREA) for img in self.data]
            
            buffer2  = [cv2.resize(img.astype('uint8'), (512,512), interpolation=cv2.INTER_AREA) for img in self.masks]
            
            self.data = buffer1
            self.masks = buffer2
            self.n_samples = len(self.data)
        

    def __len__(self): 
        return self.n_samples
    
    def _extract_bit_map(self, fname):
        with open(fname, 'r') as file:
            data = json.load(file)
        base64_string = data['objects'][0]['bitmap']['data']
        raw_bytes = base64.b64decode(base64_string)
        image_bytes = zlib.decompress(raw_bytes)
        image = Image.open(BytesIO(image_bytes))
        return np.array(image)    
    
    def _auto_mask_filter(self, image,brightness):
        h, w = image.shape[:2]
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, brightness, 255, cv2.THRESH_BINARY)
        # 2. Find all contours
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        biggest_contour = max(contours, key=cv2.contourArea)
        mask = np.zeros((h, w), dtype=np.uint8)
        # Draw filled contour in white
        mask_filter = cv2.drawContours(mask, [biggest_contour], -1, 255, thickness=cv2.FILLED)
        return mask_filter
    
    def _elastic_transform(self, image, mask, alpha=34, sigma=4):
        """
        Apply random elastic deformation (Section 3.1 of the paper).

        Parameters
        ----------
        alpha : deformation magnitude (pixels)
        sigma : smoothness of the deformation field
        """
        H, W, C = image.shape
        dx = gaussian_filter(np.random.randn(H, W), sigma) * alpha
        dy = gaussian_filter(np.random.randn(H, W), sigma) * alpha

        y_coords, x_coords = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
        indices = (np.clip(y_coords + dy, 0, H - 1).ravel(),
                np.clip(x_coords + dx, 0, W - 1).ravel())

        image_def = np.zeros_like(image)

        for i in range(C):
            image_def[...,i] = map_coordinates(image[...,i], indices, order=1).reshape(H, W)
        
        mask_def = map_coordinates(mask.astype(float), indices, order=0).reshape(H, W).astype(np.int64)
        
        return image_def, mask_def

    def _augmentation(self, image, mask):

        # Horizontal flip
        if np.random.rand() > 0.5:
            image = cv2.flip(image, 1)
            mask = cv2.flip(mask, 1)

        # Vertical flip
        if np.random.rand() > 0.5:
            image = cv2.flip(image, 0)
            mask = cv2.flip(mask, 0)

        # Random rotation
        if np.random.rand() > 0.5:
            angle = np.random.choice([90, 180, 270])

            if angle == 90:
                image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
                mask = cv2.rotate(mask, cv2.ROTATE_90_CLOCKWISE)

            elif angle == 180:
                image = cv2.rotate(image, cv2.ROTATE_180)
                mask = cv2.rotate(mask, cv2.ROTATE_180)

            elif angle == 270:
                image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
                mask = cv2.rotate(mask, cv2.ROTATE_90_COUNTERCLOCKWISE)

        return image, mask

    def __getitem__(self, idx):
        image  = self.data[idx]
        mask = self.masks[idx]

        image = image.astype(np.float32) / 255.0
            
        mask = (mask > 0).astype(np.int64)

        image = torch.tensor(image).permute(2, 0, 1).float()
        mask = torch.from_numpy(mask)
        
        return image, mask
    
def model_cehck(model, DEVICE, deep_supervission):
        dummy = torch.randn(2, 3, 256, 256).to(DEVICE)
        out = model(dummy)
        print(f'Input shape : {dummy.shape}')
        if deep_supervission:
            print(f'Output shape: {out[-1].shape}')   # should be (2, 2, 256, 256)
        else:
            print(f'Output shape: {out.shape}')   # should be (2, 2, 256, 256)
        # Quick sanity check
        total_params = sum(p.numel() for p in model.parameters())
        print(f'Total parameters: {total_params:,}')

def plot_history(history, NUM_EPOCHS, name):
    epochs = range(1, NUM_EPOCHS + 1)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].plot(epochs, history['train_loss'], label='Train', color='steelblue')
    axes[0].plot(epochs, history['val_loss'],   label='Val',   color='tomato')
    axes[0].set_title('Loss'); axes[0].set_xlabel('Epoch'); axes[0].legend()

    axes[1].plot(epochs, history['val_iou'], color='mediumseagreen', marker='o', markersize=3)
    axes[1].set_title('Validation IoU'); axes[1].set_xlabel('Epoch'); axes[1].set_ylim(0, 1)

    axes[2].plot(epochs, history['val_dice'], color='mediumpurple', marker='s', markersize=3)
    axes[2].set_title('Validation Dice'); axes[2].set_xlabel('Epoch'); axes[2].set_ylim(0, 1)

    plt.suptitle('U-Net Training History', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f"History_{name}.png")

def plot_validation(model, DEVICE, val_loader, supervission, name):
    model.eval()
    images, masks = next(iter(val_loader))
    images_gpu = images.to(DEVICE)

    with torch.no_grad():
        if supervission:
            logits = model(images_gpu)
            logits = logits[-1]
        else:
            logits = model(images_gpu)
            

    probs = F.softmax(logits, dim=1)[:, 1].cpu().numpy()  # foreground probability
    preds = logits.argmax(dim=1).cpu().numpy()

    images_np = images[:, 0].numpy()
    masks_np  = masks.numpy()

    n_show = min(4, len(images_np))
    fig, axes = plt.subplots(n_show, 4, figsize=(14, 3.5 * n_show))

    col_titles = ['Input Image', 'Ground Truth', 'Predicted Mask', 'Foreground Prob.']
    for col, title in enumerate(col_titles):
        axes[0, col].set_title(title, fontsize=11, fontweight='bold')

    for i in range(n_show):
        sample_iou  = iou_score(torch.tensor(preds[i]), torch.tensor(masks_np[i]))
        sample_dice = dice_score(torch.tensor(preds[i]), torch.tensor(masks_np[i]))

        axes[i, 0].imshow(images_np[i], cmap='gray')
        axes[i, 1].imshow(masks_np[i],  cmap='Blues', vmin=0, vmax=1)
        axes[i, 2].imshow(preds[i],     cmap='Blues', vmin=0, vmax=1)
        axes[i, 3].imshow(probs[i],     cmap='hot',   vmin=0, vmax=1)
        axes[i, 0].set_ylabel(f'IoU={sample_iou:.3f}\nDice={sample_dice:.3f}', fontsize=9)

        for ax in axes[i]: ax.axis('off')

    plt.suptitle('U-Net Predictions on Validation Set', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f"Validation_{name}.png")

def main(dirct = '.', Dataset_name = 0, model_selct = 0, Deep_vission = False, Mode = 0, Features = 32, Epochs = 20, LR = 1e-3, Batchsize = 4,AugmentSize = 5, ENHANCE = False):
    name = f"{Dataset_name}_{model_selct}_{Deep_vission}_{Mode}_{Features}_{Epochs}_{LR}_{Batchsize}_{AugmentSize}_Enhanced_{ENHANCE}"
    #Especificamos semilla para entrenamientos y reproducibilidad
    torch.manual_seed(42)
    np.random.seed(42)

    #Verificamos GPU
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {DEVICE}')

    if Mode == 0:
        if model_selct == 1:
            model = UNetpp_4(in_channels=3, out_channels=2, base_features=Features, deep_supervision=Deep_vission).to(DEVICE)
            model_cehck(model = model, DEVICE = DEVICE, deep_supervission = Deep_vission)
            model = UNetpp_4(in_channels=3, out_channels=2, base_features=Features, deep_supervision=Deep_vission).to(DEVICE)

        elif model_selct == 2:
            model = UNetpp_3(in_channels=3, out_channels=2, base_features=Features, deep_supervision=Deep_vission).to(DEVICE)
            model_cehck(model = model, DEVICE = DEVICE, deep_supervission = Deep_vission)
            model = UNetpp_3(in_channels=3, out_channels=2, base_features=Features, deep_supervision=Deep_vission).to(DEVICE)

        elif model_selct == 3:
            model = UNetpp_2(in_channels=3, out_channels=2, base_features=Features, deep_supervision=Deep_vission).to(DEVICE)
            model_cehck(model = model, DEVICE = DEVICE, deep_supervission = Deep_vission)
            model = UNetpp_2(in_channels=3, out_channels=2, base_features=Features, deep_supervision=Deep_vission).to(DEVICE)

        elif model_selct == 4:
            model = UNetpp_1(in_channels=3, out_channels=2, base_features=Features, deep_supervision=Deep_vission).to(DEVICE)
            model_cehck(model = model, DEVICE = DEVICE, deep_supervission = Deep_vission)
            model = UNetpp_1(in_channels=3, out_channels=2, base_features=Features, deep_supervision=Deep_vission).to(DEVICE)

        else:
            print("No funciona actualemente - outputs dimensions wrong ??????")
            model = UNet(in_channels=3, out_channels=2, base_features=Features).to(DEVICE)
            model_cehck(model = model, DEVICE = DEVICE, deep_supervission = Deep_vission)
            model = UNet(in_channels=3, out_channels=2, base_features=Features).to(DEVICE)
            

        print(f'Acquiring Dataset...')
        train_dataset = veinDataset(dir = dirct, training=True, augment=True, Drive = Dataset_name, Actor =1, augment_amount = AugmentSize, ENHANCE = ENHANCE)
        val_dataset   = veinDataset(dir = dirct, training=False, Drive = Dataset_name, Actor =1,  augment=False, ENHANCE = ENHANCE)

        train_loader = DataLoader(train_dataset, batch_size=Batchsize, shuffle=True,  num_workers=0)
        val_loader   = DataLoader(val_dataset,   batch_size=Batchsize, shuffle=False, num_workers=0)
        print(f'Dataset Acquired')

        optimizer = torch.optim.Adam(model.parameters(), lr=LR)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)
        model, History = Training(model = model, scheduler = scheduler, NUM_EPOCHS = Epochs, DEVICE = DEVICE, deep_supervission = Deep_vission, train_loader = train_loader, val_loader = val_loader, optimizer = optimizer, name = name)
        
        plot_history(history = History, NUM_EPOCHS = Epochs, name = name)
        plot_validation(model = model, DEVICE = DEVICE, val_loader = val_loader, supervission = Deep_vission, name = name)

    else:
        if model_selct == 1:
            model = UNetpp_4(in_channels=3, out_channels=2, base_features=Features, deep_supervision=Deep_vission).to(DEVICE)
            model.load_state_dict(torch.load('./pp4/model_weights_pp4.pth', weights_only=True))
            model.eval()
        elif model_selct == 2:
            model = UNetpp_3(in_channels=3, out_channels=2, base_features=Features, deep_supervision=Deep_vission).to(DEVICE)
            model.load_state_dict(torch.load('./pp3/model_weights_pp3.pth', weights_only=True))
            model.eval()
        elif model_selct == 3:
            model = UNetpp_2(in_channels=3, out_channels=2, base_features=Features, deep_supervision=Deep_vission).to(DEVICE)
            model.load_state_dict(torch.load('./pp2/model_weights_pp2.pth', weights_only=True))
            model.eval()
        elif model_selct == 4:
            model = UNetpp_1(in_channels=3, out_channels=2, base_features=Features, deep_supervision=Deep_vission).to(DEVICE)
            model.load_state_dict(torch.load('./pp1/model_weights_pp1.pth', weights_only=True))
            model.eval()
        else:
            print("No implementado")
            model = UNet(in_channels=3, out_channels=2, base_features=Features).to(DEVICE)
            model.load_state_dict(torch.load('model_weights.pth', weights_only=True))
            model.eval()

        if Dataset_name == 2:
            val_dataset   = veinDataset(dir = dirct, training=False, Drive = Dataset_name, Actor =1,  augment=False, ENHANCE = ENHANCE)
            val_loader    = DataLoader(val_dataset,   batch_size=Batchsize, shuffle=False, num_workers=0)
            predictions = []
            probs_list = []
            masks_list = []

            for image, masks in iter(val_loader):        
                images_gpu = image.to(DEVICE)
                with torch.no_grad():
                    if Deep_vission:
                        logits = model(images_gpu)
                        preds = logits[-1].argmax(dim=1).cpu().numpy()              
                        probs = F.softmax(logits[-1], dim=1)[:, 1].cpu().numpy() 
                    else:
                        logits = model(images_gpu)
                        preds = logits.argmax(dim=1).cpu().numpy()
                        probs = F.softmax(logits, dim=1)[:, 1].cpu().numpy() 
                probs_list.append(probs)
                predictions.append(preds)
                masks_list.append(masks.cpu().numpy())

            _, summary = dataset_metrics(masks_list,predictions,probs_list)    
            
            with open(f"Summary_{name}.json", "w") as f:
                json.dump(summary, f)

            plot_validation(model = model, DEVICE = DEVICE, val_loader = val_loader, supervission = Deep_vission, name = name)

        elif Dataset_name == 1:
            val_dataset   = veinDataset(dir = dirct, training=False, Drive = Dataset_name, Actor =1,  augment=False, ENHANCE = ENHANCE)
            val_loader    = DataLoader(val_dataset,   batch_size=Batchsize, shuffle=False, num_workers=0)
        
            predictions = []
            probs_list = []
            masks_list = []

            for image, masks in iter(val_loader):        
                images_gpu = image.to(DEVICE)
                with torch.no_grad():
                    if Deep_vission:
                        logits = model(images_gpu)
                        preds = logits[-1].argmax(dim=1).cpu().numpy()              
                        probs = F.softmax(logits[-1], dim=1)[:, 1].cpu().numpy() 
                    else:
                        logits = model(images_gpu)
                        preds = logits.argmax(dim=1).cpu().numpy()
                        probs = F.softmax(logits, dim=1)[:, 1].cpu().numpy() 
                probs_list.append(probs)
                predictions.append(preds)
                masks_list.append(masks.cpu().numpy())

            _, summary = dataset_metrics(masks_list,predictions,probs_list)    
            with open(f"Summary_{name}_Actor_1.json", "w") as f:
                json.dump(summary, f)          
            plot_validation(model = model, DEVICE = DEVICE, val_loader = val_loader, supervission = Deep_vission, name = name+"_Autor_1")

            val_dataset   = veinDataset(dir = dirct, training=False, Drive = Dataset_name, Actor = 2,  augment=False, ENHANCE = ENHANCE)
            val_loader    = DataLoader(val_dataset,   batch_size=Batchsize, shuffle=False, num_workers=0)
        
            predictions = []
            probs_list = []
            masks_list = []

            for image, masks in iter(val_loader):        
                images_gpu = image.to(DEVICE)
                with torch.no_grad():
                    if Deep_vission:
                        logits = model(images_gpu)
                        preds = logits[-1].argmax(dim=1).cpu().numpy()              
                        probs = F.softmax(logits[-1], dim=1)[:, 1].cpu().numpy() 
                    else:
                        logits = model(images_gpu)
                        preds = logits.argmax(dim=1).cpu().numpy()
                        probs = F.softmax(logits, dim=1)[:, 1].cpu().numpy() 
                probs_list.append(probs)
                predictions.append(preds)
                masks_list.append(masks.cpu().numpy())

            _, summary = dataset_metrics(masks_list,predictions,probs_list)     
            with open(f"Summary_{name}_Actor_2.json", "w") as f:
                json.dump(summary, f) 

            plot_validation(model = model, DEVICE = DEVICE, val_loader = val_loader, supervission = Deep_vission, name = name+"_Autor_2")
                                      
        else:

            val_dataset   = veinDataset(dir = dirct, training=False, Drive = Dataset_name, Actor =1,  augment=False, ENHANCE = ENHANCE)
            val_loader    = DataLoader(val_dataset,   batch_size=Batchsize, shuffle=False, num_workers=0)
        
            predictions = []
            probs_list = []
            masks_list = []

            for image, masks in iter(val_loader):        
                images_gpu = image.to(DEVICE)
                with torch.no_grad():
                    if Deep_vission:
                        logits = model(images_gpu)
                        preds = logits[-1].argmax(dim=1).cpu().numpy()              
                        probs = F.softmax(logits[-1], dim=1)[:, 1].cpu().numpy() 
                    else:
                        logits = model(images_gpu)
                        preds = logits.argmax(dim=1).cpu().numpy()
                        probs = F.softmax(logits, dim=1)[:, 1].cpu().numpy() 
                probs_list.append(probs)
                predictions.append(preds)
                masks_list.append(masks.cpu().numpy())

            _, summary = dataset_metrics(masks_list,predictions,probs_list)    
            with open(f"Summary_{name}.json", "w") as f:
                json.dump(summary, f)

            plot_validation(model = model, DEVICE = DEVICE, val_loader = val_loader, supervission = Deep_vission, name = name)



if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--Database",
        type=str,
        required=True,
        help="Root Directory for database, def: Current Directory ",
        default='.'
    )
    parser.add_argument(
        "--Dataset_name",
        type=int,
        required=True,
        help="Name: 0 - DRIVE (Training), 1 - STARE (Concordancia), 2 - CHASE_DB1 (Comparacion), def: 0",
        default=0
    )
    parser.add_argument(
        "--Model",
        type=int,
        required=True,
        help="Model Selecction: 1 - Unet++(4), 2 - Unet++(3), 3 - Unet++(2), 4 -  Unet++(1), def: 1",
        default=1
    )
    parser.add_argument(
        "--DeepSV",
        type=bool,
        required=False,
        help="Deep Supervission (True - False), def: False",
        default=False
    )
    parser.add_argument(
        "--Mode",
        type=int,
        required=True,
        help="Mode: 0 - Training, 1 - validation, def: 0",
        default=0
    )
    parser.add_argument(
        "--Features",
        type=int,
        required=False,
        help="Feature quantity for UNet Model, def: 32",
        default=32
    )
    parser.add_argument(
        "--Epochs",
        type=int,
        required=False,
        help="Number of Epochs for traning, def: 20",
        default=20
    )

    parser.add_argument(
        "--LR",
        type=float,
        required=False,
        help="Learning Rate Selection, def: 1e-3",
        default=1e-3
    )    
    parser.add_argument(
        "--BatchSize",
        type=int,
        required=False,
        help="batch size, def: 4",
        default=4
    )       
    parser.add_argument(
        "--AugmentSize",
        type=int,
        required=False,
        help="Times Data will be augmented, def: 5",
        default=5
    )      
    parser.add_argument(
        "--Enhance_images",
        type=bool,
        required=False,
        help="Enhance image brightness and contrast (works for databases 1 and 2 - NO TRAINNING), def: False",
        default=False
    )         

    args = parser.parse_args()


    main(
        args.Database,
        args.Dataset_name,
        args.Model,
        args.DeepSV,
        args.Mode,
        args.Features,
        args.Epochs,
        args.LR,
        args.BatchSize,
        args.AugmentSize,
        args.Enhance_images
    )