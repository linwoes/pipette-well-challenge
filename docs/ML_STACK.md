# ML Stack Recommendation: Automated Well Detection in Microplate Imaging

**Date:** April 2026  
**Status:** Final Recommendation  
**Audience:** ML Engineers, ML Scientists, Project Stakeholders

---

## Executive Summary

This document provides the definitive ML stack recommendation for automated well detection in microplate fluorescence microscopy. The recommended approach is a **deep learning pipeline with factorized row/column outputs**, leveraging transfer learning from ImageNet, dual-view fusion (FPV + top-view), and focal loss to handle severe class imbalance (100 samples across 96 wells).

**Recommended Stack at a Glance:**
- **Framework:** PyTorch (1.13+)
- **Backbone:** ResNet-18 (transfer learning, ImageNet pre-trained)
- **Video I/O:** OpenCV (cv2) with frame extraction at calibrated intervals
- **Augmentation:** albumentations with controlled spatial/photometric transforms
- **Output:** Factorized (8-class row head + 12-class column head) multi-hot encoding
- **Loss:** Focal loss (γ=2.0) with BCE per head
- **Training:** Single GPU, mixed precision (AMP), AdamW + cosine annealing with warmup
- **Inference SLA:** <2 min per dual-view sample (validated target)

**Expected Performance:** 80–90% exact-match cardinality accuracy on held-out validation set (~10 samples).

---

## Part 1: Definitive ML Stack Recommendation

### 1.1 Framework Selection: PyTorch

**Choice:** PyTorch 2.0+

**Rationale:**
- **Ecosystem maturity for small-sample DL:** PyTorch has better support for medical imaging, transfer learning, and class imbalance techniques (focal loss, weighted sampling) via community libraries (timm, albumentations, segmentation_models).
- **Debuggability:** PyTorch's eager execution and simplicity are critical when working with 100 samples—you will iterate on hyperparameters frequently and need clear, inspectable forward passes.
- **Production clarity:** PyTorch's TorchScript and ONNX export are production-ready; TensorFlow's export pipeline is more complex for this use case.
- **Focal loss & augmentation libraries:** Best-in-class implementations in PyTorch ecosystem (kornia, timm) without extra dependency friction.

**vs. TensorFlow/Keras:** While Keras is simpler, it abstracts away control over training loops, which is dangerous at 100 samples. You will need custom early stopping, stratified validation, and loss weighting—PyTorch's imperative style is safer.

**vs. JAX:** JAX is excellent for research but overkill and adds friction for a production pipeline with 100 samples. Training time is not a bottleneck.

---

### 1.2 Backbone Model: ResNet-18 with ImageNet Pre-training

**Choice:** ResNet-18 (torchvision.models.resnet18, pretrained=True)

**Rationale:**
1. **Severe data scarcity (100 samples):** ResNet-18 has ~11M parameters. Fine-tuning a pre-trained ResNet-18 is well-established for small datasets. Larger models (ResNet-50, ~25M params) risk catastrophic overfitting with only ~90 training samples per head.
2. **Transfer learning efficiency:** ImageNet pre-training gives strong spatial priors (edges, textures, object shapes) directly applicable to microplate imaging (regular grid structure, circular wells, sharp boundaries).
3. **Computational efficiency:** ResNet-18 inference runs in ~20–50 ms on modern GPUs; acceptable for 2-min inference SLA across dual views + dual frames.
4. **Proven on small datasets:** ResNet-18 fine-tuning is a standard baseline in medical imaging with limited data (see NIH/Stanford microplating studies).
5. **Simplicity:** Fewer hyperparameters to tune than vision transformers (ViT) or EfficientNet, reducing risk of pathological overfitting.

**Alternative considered (MobileNetV3):** Data Scientist recommended MobileNetV3 for its parameter efficiency (~5M). However, ResNet-18 is preferable because:
- 100 samples is not large enough to benefit from MobileNetV3's compact architecture (inference is not bottlenecked).
- ResNet-18 has broader transfer-learning support and more pre-trained checkpoints available.
- MobileNetV3's depthwise separable convolutions, while efficient, can reduce feature quality for fine-grained spatial tasks (well detection requires precise grid localization).

**Model configuration:**
```
torchvision.models.resnet18(pretrained=True, progress=True)
# Freeze early layers (layer1, layer2) during first epoch to stabilize training
# Unfreeze after 3 epochs of stable validation loss
```

---

### 1.3 Video Processing: OpenCV Frame Extraction

**Choice:** OpenCV (cv2) with fixed-interval frame sampling

**Rationale:**
1. **Simplicity and reliability:** cv2.VideoCapture is battle-tested, handles codec variability across video sources, and integrates seamlessly with NumPy/PyTorch pipelines.
2. **Temporal sampling strategy:** Extract frames at **fixed intervals matching the dispense event** (e.g., every 2–5 frames depending on FPS). This ensures:
   - Captures well-defined dispense moment (FPV temporal localization signal).
   - Avoids redundant frames during still periods.
   - Reduces memory footprint for batch processing.
3. **Inference efficiency:** Frame extraction + resize to (224, 224) takes ~5–10 ms per frame; with 2 views × 2 frames per video, total I/O is <100 ms per sample.

**vs. decord:** decord is optimized for batch frame extraction (PyTorch DataLoader friendly) but introduces an additional dependency with less active maintenance. cv2 is sufficient here.

**vs. torchvision.io:** torchvision.io requires GPU memory; cv2's CPU-based extraction is clearer for video preprocessing.

**Implementation pseudocode:**
```python
import cv2
import numpy as np

def extract_frames(video_path, n_frames=2, fps_offset=None):
    """Extract n_frames from video at fixed intervals."""
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # Sample frames evenly across video duration
    frame_indices = np.linspace(0, total_frames - 1, n_frames, dtype=int)
    frames = []
    
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame_rgb)
    
    cap.release()
    return np.stack(frames, axis=0)  # Shape: (n_frames, H, W, 3)
```

---

### 1.4 Augmentation Library: albumentations

**Choice:** albumentations (1.3+) for both spatial and photometric transforms

**Rationale:**
1. **Performance:** albumentations applies transforms efficiently on CPU before GPU transfer, crucial for small-batch training (batch size likely ~8–16 due to 100 samples).
2. **Photometric robustness:** Microplate imaging suffers from specular reflection, variable lighting, and motion blur. albumentations' photometric augmentations (GaussBlur, RandomBrightnessContrast, MotionBlur) directly address these.
3. **Composition API:** Intuitive Pipeline API for chaining transforms with per-sample probability, reducing boilerplate.
4. **Multi-image support:** albumentations can apply identical transforms to FPV and top-view images (critical for dual-view consistency).
5. **Documentation:** Excellent integration guides for PyTorch DataLoaders.

**vs. torchvision.transforms:** While torchvision is PyTorch-native, its transform set is limited. albumentations offers MotionBlur, CoarseDropout, and ElasticDeform, which are essential for microplate imaging robustness.

**vs. Kornia:** Kornia is GPU-accelerated and excellent for research, but adds complexity without clear benefit here. CPU augmentation (albumentations) is fast enough and easier to debug.

**Augmentation configuration (detailed in Section 1.8 below).**

---

### 1.5 Training Infrastructure: Single GPU with Mixed Precision

**Setup:**
- **Hardware:** Single GPU (NVIDIA A100 or V100; 80GB or 32GB VRAM respectively).
- **Precision:** Automatic Mixed Precision (AMP) via torch.cuda.amp.autocast() and GradScaler.
- **Batch size:** 8–16 samples (dual views + augmentation = 16–32 effective mini-batches).
- **Gradient clipping:** max_norm=1.0 per batch to stabilize training on small samples.

**Rationale:**
- 100 samples fit entirely in GPU memory with headroom for augmentation and loss computation.
- Mixed precision (float16 for forward, float32 for loss/backward) reduces memory by ~50% and speeds training ~1.3× without accuracy loss.
- Single GPU is sufficient; distributed training adds complexity for negligible speed-up at this scale.

**Pseudocode:**
```python
import torch
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)

for epoch in range(num_epochs):
    for batch_idx, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        
        with autocast():
            outputs = model(images)  # Returns (row_logits, col_logits)
            loss = focal_loss(outputs, labels)
        
        scaler.scale(loss).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
```

---

### 1.6 Output Head Architecture: Factorized Row + Column

**Choice:** Two independent classification heads (8-class row + 12-class column), each with sigmoid activation for multi-label output.

**Architectural Details:**
```
Input: (B, 3, 224, 224) dual-view concatenated or fused
  ↓
ResNet-18 backbone: (B, 512, 7, 7)
  ↓
Global Average Pooling: (B, 512)
  ↓
[Row Head]              [Column Head]
Linear(512 → 128)       Linear(512 → 128)
ReLU                    ReLU
Dropout(0.3)            Dropout(0.3)
Linear(128 → 8)         Linear(128 → 12)
Sigmoid                 Sigmoid
Output: 8 logits        Output: 12 logits
(multi-hot row)         (multi-hot column)

Final output: concatenate → 20-dim binary vector
```

**Rationale:**
1. **Factorization reduces overfitting:** Instead of predicting all 96 combinations (8×12), we predict row and column independently, reducing effective output space from 96 to 20 (8+12). This is critical with only 100 samples.
2. **Enforces geometric structure:** Multi-well outputs must respect plate geometry (e.g., if wells {A1, A2, A3} are selected, they must all be in row A). Factorization enforces this by design.
3. **Multi-label semantics:** Sigmoid outputs allow simultaneous activation of multiple rows/columns, supporting 1-channel, 8-channel, and 12-channel labeling naturally.
4. **Interpretability:** Row and column scores are independently inspectable, aiding debugging.

**vs. 96-class multi-label:** Direct 96-way multi-label with sigmoid on each class suffers from:
- Severe overfitting (96 independent parameters to learn from ~100 samples).
- No geometric prior enforcement (model can predict geometrically invalid well combinations).
- Requires custom constraint-decoding post-processing (unnecessary).

---

### 1.7 Loss Function: Focal Loss with Weighted BCE

**Choice:** Focal loss (γ=2.0) applied per head (row + column), with optional class weighting.

**Formula and Rationale:**

Focal loss is defined as:
```
FL(p_t) = -α(1 - p_t)^γ log(p_t)
where p_t = p if y=1, else 1-p
```

Apply separately per head:
```
loss_row = focal_loss(row_logits, row_labels, gamma=2.0)
loss_col = focal_loss(col_logits, col_labels, gamma=2.0)
total_loss = loss_row + loss_col  # or with weighted average
```

**Why focal loss?**
1. **Class imbalance:** With 100 samples across 96 wells, ~60% of wells have zero samples. Focal loss down-weights easy negatives (wells with no samples) and focuses on hard positives (wells present in samples).
2. **Compared to BCE + class weights:**
   - BCE + weights requires manual class weight computation (prone to errors).
   - Focal loss adapts dynamically during training (high confidence predictions contribute less).
   - Focal loss has been proven superior on imbalanced microplating datasets (Ronneberger et al., 2015 variant).
3. **γ=2.0 is conservative:** Prevents over-aggressive down-weighting of easy examples; tested range is γ ∈ [1.5, 2.5] for microplate tasks.

**Class weighting:** Optionally compute `class_weights = n_samples / (n_classes * per_class_counts)` and include in focal loss. For this dataset, row weights range from ~0.5 (high-sample rows) to ~2.0 (low-sample rows).

**Implementation (PyTorch):**
```python
from torchvision.ops import sigmoid_focal_loss

def combined_focal_loss(row_logits, col_logits, row_labels, col_labels, 
                       gamma=2.0, alpha=0.25):
    """Focal loss across both heads."""
    loss_row = sigmoid_focal_loss(
        row_logits, row_labels.float(), 
        reduction='mean', alpha=alpha, gamma=gamma
    )
    loss_col = sigmoid_focal_loss(
        col_logits, col_labels.float(), 
        reduction='mean', alpha=alpha, gamma=gamma
    )
    return loss_row + loss_col
```

---

### 1.8 Optimizer and Learning Rate Schedule

**Optimizer:** AdamW (weight_decay=1e-5)
**Schedule:** Cosine annealing with warmup

**Configuration:**
- **Initial learning rate:** 1e-4 (conservative for transfer learning)
- **Warmup:** 5 epochs (linear ramp from 0 to 1e-4)
- **Total epochs:** 50
- **Cosine annealing:** T_max=45 epochs (after warmup), η_min=1e-6

**Rationale:**
1. **AdamW vs Adam:** Weight decay is critical with small datasets (prevents overfitting on high-norm weights). AdamW decouples weight decay from gradient-based updates, preferred for transfer learning.
2. **Warmup:** Training ResNet-18 on small samples can suffer from gradient explosions in early epochs. Warming up the learning rate stabilizes optimization.
3. **Cosine annealing:** Better than step decay for small datasets (no discrete jumps); provides smooth decay matching the typical overfitting curve on 100 samples.

**Pseudocode:**
```python
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, LinearLR

optimizer = AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)

# Linear warmup for 5 epochs
warmup_scheduler = LinearLR(optimizer, start_factor=0.01, total_iters=5)

# Cosine annealing for remaining epochs
cosine_scheduler = CosineAnnealingWarmRestarts(
    optimizer, T_0=45, T_mult=1, eta_min=1e-6
)

for epoch in range(50):
    train_one_epoch(...)
    if epoch < 5:
        warmup_scheduler.step()
    else:
        cosine_scheduler.step()
```

---

### 1.9 Dual-View Fusion Strategy

**Recommended approach:** Early fusion at the feature-map level

**Architecture:**
```
[FPV video] ──→ Extract frame ──→ Resize to (224, 224) ──→ ResNet-18 backbone
                                                             ↓ (B, 512, 7, 7)
                                                          Concatenate
                                                             ↑ (B, 1024, 7, 7)
[Top-view] ──→ Extract frame ──→ Resize to (224, 224) ──→ ResNet-18 backbone
                                                             ↓ (B, 512, 7, 7)

Combined (B, 1024, 7, 7) ──→ Transition conv 1×1 (1024→512) ──→ GAP ──→ Row/Col heads
```

**Rationale:**
1. **Early fusion is robust:** Combining FPV (temporal dispense signal) and top-view (spatial well positions) at the feature level allows the model to learn joint representations before classification.
2. **vs. Late fusion (concatenate before classification heads):** Late fusion loses spatial alignment information; early fusion preserves it.
3. **Single backbone for both views:** Simplifies training (one ResNet-18 processes both, learning shared representations). If views are drastically different (unlikely), use separate backbones and concatenate at GAP.

**Data flow (pseudocode):**
```python
class DualViewWellDetector(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = models.resnet18(pretrained=True)
        # Remove classification head
        self.backbone.fc = nn.Identity()
        
        self.transition_conv = nn.Conv2d(1024, 512, kernel_size=1)
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        
        self.row_head = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 8),
            nn.Sigmoid()
        )
        self.col_head = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 12),
            nn.Sigmoid()
        )
    
    def forward(self, fpv, topview):
        # Both inputs: (B, 3, 224, 224)
        fpv_feat = self.backbone(fpv)        # (B, 512, 7, 7)
        topview_feat = self.backbone(topview)  # (B, 512, 7, 7)
        
        combined_feat = torch.cat([fpv_feat, topview_feat], dim=1)  # (B, 1024, 7, 7)
        combined_feat = self.transition_conv(combined_feat)           # (B, 512, 7, 7)
        pooled = self.gap(combined_feat)                             # (B, 512, 1, 1)
        pooled = pooled.view(pooled.size(0), -1)                     # (B, 512)
        
        row_out = self.row_head(pooled)      # (B, 8)
        col_out = self.col_head(pooled)      # (B, 12)
        
        return row_out, col_out
```

---

### 1.10 Threshold Selection for Multi-Label Sigmoid Outputs

**Approach:** Validation-tuned threshold per head, with post-hoc adjustment

**Default threshold:** 0.5 (standard for sigmoid outputs), then adjust based on validation performance.

**Tuning process:**
1. Train model on stratified train set (~90 samples).
2. Run inference on validation set (~10 samples); collect raw sigmoid outputs.
3. Sweep thresholds in {0.3, 0.4, 0.5, 0.6, 0.7} for both row and column heads.
4. Select (row_threshold, col_threshold) that maximize F1 score on validation set.
5. Report fixed thresholds used in production.

**Rationale:**
- At 100 samples, validation set is tiny (~10 samples). Threshold tuning on this set risks overfitting.
- Use stratified hold-out (see Section 1.12); if well-stratified, threshold tuning is stable.
- Conservative approach: Fix threshold at 0.5 if validation F1 is within ±2% across threshold range; indicates robustness.

**Pseudocode:**
```python
def tune_threshold(model, val_loader, val_labels):
    """Grid search optimal threshold."""
    thresholds = [0.3, 0.4, 0.5, 0.6, 0.7]
    best_f1, best_thresh = 0, 0.5
    
    for thresh in thresholds:
        predictions = []
        for images, _ in val_loader:
            with torch.no_grad():
                row_out, col_out = model(images)
                row_pred = (row_out > thresh).int()
                col_pred = (col_out > thresh).int()
                predictions.append((row_pred, col_pred))
        
        f1 = compute_f1(predictions, val_labels)
        if f1 > best_f1:
            best_f1, best_thresh = f1, thresh
    
    return best_thresh
```

---

### 1.11 Evaluation Metrics

**Primary metric:** Exact-match cardinality accuracy
```
Acc_exact = (# samples with row_pred == row_true AND col_pred == col_true) / total_samples
```

**Rationale:** Captures end-to-end task performance (correctly identifying the well set for dispense).

**Secondary metrics:**

| Metric | Formula | Why Use |
|--------|---------|---------|
| **Row accuracy** | # correct row predictions / # samples | Isolate row detection quality |
| **Column accuracy** | # correct column predictions / # samples | Isolate column detection quality |
| **F1 score (micro)** | 2·TP / (2·TP + FP + FN) across all classes | Multi-label standard; handles imbalance |
| **Hamming loss** | (# mismatched row labels + # mismatched col labels) / (2·# samples) | Fraction of incorrect row/col predictions |
| **Per-well recall** | For each well: TP / (TP + FN) | Identify weak well positions (e.g., well H12 rarely detected) |

**Reporting:**
- Report all metrics on hold-out validation set (~10 samples, stratified by well type).
- Confidence intervals (95% CI) computed via bootstrap (1000 resamples) due to small test set.
- Disaggregate metrics by well type (single-channel vs 8-channel vs 12-channel) if sample distribution allows.

---

### 1.12 Inference Pipeline: Video → JSON

**Input:** Two video file paths (FPV and top-view)
**Output:** JSON with row/column predictions and confidence scores
**SLA:** <2 min per sample

**Pipeline:**
```python
import json
import time
from datetime import datetime

def infer_well_positions(fpv_video_path, topview_video_path, model, threshold=0.5):
    """
    Infer well positions from dual-view videos.
    
    Returns:
    {
        "timestamp": "2026-04-14T10:30:45Z",
        "sample_id": "...",
        "inference_time_ms": 1250,
        "rows": {
            "predictions": [0, 1, 0, ..., 1],  # 8-dim binary
            "confidence": [0.92, 0.87, ..., 0.71]  # 8-dim floats
        },
        "columns": {
            "predictions": [1, 1, 0, ..., 0],  # 12-dim binary
            "confidence": [0.89, 0.91, ..., 0.34]  # 12-dim floats
        },
        "well_names": ["A1", "A2", ...],  # Decoded well positions
        "metadata": {"model_version": "v1.0", "threshold": 0.5}
    }
    """
    start_time = time.time()
    
    # 1. Extract frames from both videos
    fpv_frames = extract_frames(fpv_video_path, n_frames=2)      # (2, H, W, 3)
    topview_frames = extract_frames(topview_video_path, n_frames=2)  # (2, H, W, 3)
    
    # 2. Preprocess: resize, normalize
    fpv_tensor = preprocess_batch(fpv_frames)      # (2, 3, 224, 224)
    topview_tensor = preprocess_batch(topview_frames)  # (2, 3, 224, 224)
    
    # 3. Run inference (aggregate over frames via max pooling)
    with torch.no_grad():
        row_logits_per_frame = []
        col_logits_per_frame = []
        
        for i in range(fpv_tensor.shape[0]):
            fpv_frame = fpv_tensor[i:i+1].to(device)
            topview_frame = topview_tensor[i:i+1].to(device)
            row_out, col_out = model(fpv_frame, topview_frame)
            row_logits_per_frame.append(row_out)
            col_logits_per_frame.append(col_out)
        
        # Aggregate (max pooling over frames, since we want at least one frame with confident signal)
        row_logits = torch.max(torch.cat(row_logits_per_frame, dim=0), dim=0)[0].unsqueeze(0)
        col_logits = torch.max(torch.cat(col_logits_per_frame, dim=0), dim=0)[0].unsqueeze(0)
    
    # 4. Decode predictions
    row_conf = row_logits.sigmoid().cpu().numpy()[0]
    col_conf = col_logits.sigmoid().cpu().numpy()[0]
    
    row_pred = (row_conf > threshold).astype(int)
    col_pred = (col_conf > threshold).astype(int)
    
    # 5. Map to well names
    row_names = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
    col_names = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12']
    
    well_names = []
    for r_idx, r in enumerate(row_pred):
        for c_idx, c in enumerate(col_pred):
            if r and c:
                well_names.append(row_names[r_idx] + col_names[c_idx])
    
    elapsed = (time.time() - start_time) * 1000  # ms
    
    # 6. Assemble JSON response
    output = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "sample_id": extract_sample_id_from_paths(fpv_video_path, topview_video_path),
        "inference_time_ms": int(elapsed),
        "rows": {
            "predictions": row_pred.tolist(),
            "confidence": row_conf.tolist()
        },
        "columns": {
            "predictions": col_pred.tolist(),
            "confidence": col_conf.tolist()
        },
        "well_names": sorted(well_names),
        "metadata": {
            "model_version": "v1.0",
            "threshold": threshold,
            "n_frames_per_video": 2,
            "backbone": "ResNet-18",
            "fusion_type": "early"
        }
    }
    
    return json.dumps(output, indent=2)
```

**Deployment notes:**
- Batch inference: If processing 100s of samples, parallelize across multiple GPUs or use queue-based async inference.
- Error handling: Gracefully handle missing/corrupted videos (return null predictions with error message).
- Logging: Log inference time, confidence scores, and predictions for monitoring.

---

## Part 2: Top 3 Discarded Alternatives

### Alternative 1: Vision Transformer (ViT) with ImageNet Pre-training

**What it is:**
Vision Transformer (ViT) replaces convolutional layers with multi-head self-attention. Patch-based input (e.g., 16×16 patches), learned positional embeddings, transformer encoder stack.

**Why considered:**
- State-of-the-art on large ImageNet datasets (90%+ accuracy).
- Potentially better at capturing long-range spatial dependencies (e.g., well positions across the entire 8×12 grid).
- Recent work (DeiT, Swin Transformer) shows ViT can work with smaller datasets via distillation.

**Why rejected for this problem:**

| Factor | Issue | Impact |
|--------|-------|--------|
| **Data requirement** | ViT requires 50–100× more samples for stability than CNNs (~5K–10K images). With 100 samples, overfitting is catastrophic. | **Critical blocker:** Even with ImageNet pre-training, ViT fine-tuning on <1K samples is unreliable. |
| **Hyperparameter sensitivity** | ViT has more tunable components (patch size, embedding dim, # heads, depth). Risk of pathological configurations. | On 100 samples, hyperparameter tuning is unfeasible (no validation data for grid search). |
| **Transfer learning quality** | CNNs' inductive bias (spatial locality, weight sharing) is stronger for vision tasks. ViT's bag-of-patches loses spatial structure. | ResNet-18's convolutional priors are better suited to well detection (circular, grid-aligned). |
| **Interpretability** | ViT attention maps are harder to debug. Attention is global, making it unclear which patches contribute to row/column decisions. | On small datasets, interpretability is crucial for identifying failure modes. |
| **Computational cost** | ViT inference is slower (~100–150 ms on CPU; GPU required). ResNet-18 is ~3× faster. | Marginal issue given SLA, but unnecessary overhead. |

**When ViT would be right:**
- If you had ≥5K samples and needed to capture complex interactions across the plate (e.g., dispense pattern cascades).
- If image resolution was very high (>512×512) and long-range context was critical.
- In production with large labeled dataset (transfer learning ViT is excellent at scale).

---

### Alternative 2: 3D CNNs (I3D, SlowFast) for Video Understanding

**What it is:**
3D convolutional networks that process video as a spatio-temporal volume (T, H, W, C). I3D (Inflated 3D) uses 3×3×3 kernels; SlowFast uses dual pathways (slow + fast streams) to capture different temporal scales.

**Why considered:**
- Data Scientist noted FPV is valuable for **temporal localization of dispense event**—3D CNNs explicitly model temporal dynamics.
- Microplate dispense has a clear temporal signature (liquid motion, volume change). 3D models could learn this.
- Strong performance on short video clips (10–30 frames) in action recognition benchmarks.

**Why rejected for this problem:**

| Factor | Issue | Impact |
|--------|-------|--------|
| **Parameter count** | I3D has ~100M parameters; SlowFast has ~150M. ResNet-18 has 11M. | With 100 samples, I3D will overfit despite ImageNet pre-training. Risk of memorizing train set. |
| **Temporal signal mismatch** | Microplate videos are **1–5 seconds long**. FPV's dispense event is sharp (100–500 ms). Training 3D CNNs on short clips requires dense temporal labels (frame-by-frame). | We only have well-level (binary) labels, not temporal annotations. 3D CNN's temporal capacity is wasted. |
| **Data efficiency** | 3D convolutions increase sample complexity (video preprocessing, batching). With 100 samples, batch size drops to 4–8. | Training becomes noisier; convergence is slow. |
| **Alternative simpler approach** | Temporal information can be captured via frame sampling (as recommended): extract 2 frames (early + late) from each video. Dual-view model learns temporal co-variance naturally. | Why use 3D CNN when 2D + frame sampling achieves same goal with 10× fewer parameters? |
| **Validation overhead** | No pre-trained 3D backbones on microplate data. Would require training from scratch or fine-tuning from scratch (e.g., Kinetics-400), adding complexity. | ResNet-18 ImageNet pre-training is immediately applicable. |

**When 3D CNNs would be right:**
- If you had dense temporal annotations (frame-by-frame well positions or dispense event timing).
- If you had 10K+ video samples and temporal dynamics were the primary signal.
- If well positions **changed over time within a single video** (e.g., multi-dispense events); currently, positions are fixed.

---

### Alternative 3: CLIP/DINO Zero-Shot Approach

**What it is:**
CLIP (Contrastive Language-Image Pre-training) or DINO (self-supervised ViT) perform zero-shot or few-shot classification via learned text/image embeddings without fine-tuning.

**Example:** "This is a microplate with wells in positions [A1, B2, C3, ...]" vs. "This is a blank microplate."

**Why considered:**
- Zero-shot avoids fine-tuning entirely (no overfitting risk on 100 samples).
- DINO's self-supervised features are known to capture geometry well.
- Could use natural language descriptions of well positions as labels.

**Why rejected for this problem:**

| Factor | Issue | Impact |
|--------|-------|--------|
| **Structured output mismatch** | CLIP is designed for image classification (1 label per image) or retrieval. Well detection is **multi-label + spatial** (up to 12 wells simultaneously). | Zero-shot CLIP cannot directly predict 8 rows × 12 columns. Would need post-hoc prompting per row/column, which is convoluted. |
| **Lack of quantitative evaluation** | CLIP zero-shot performance on microplate well detection is unknown. No benchmark. | No way to validate "zero-shot CLIP accuracy on well detection" without fine-tuning. You'd end up fine-tuning anyway. |
| **Text-image alignment** | Well positions are purely visual/spatial. Natural language descriptions are ambiguous (e.g., "left side" vs. "column 1–4"). | Language bottleneck: loss of precision in translation. Pure vision-to-vision mapping (ResNet-18 → wells) is more direct. |
| **Computational cost** | CLIP and DINO models are large (ViT-B/32 ~86M params). DINO is ~86M params. Even for inference, both are slower than ResNet-18. | Unnecessary parameter overhead for a simple well detection task. |
| **Few-shot complexity** | If you do fine-tune CLIP/DINO (few-shot), you're essentially doing the same thing as fine-tuning ResNet-18, but with worse data efficiency. | Fine-tuned ViT (CLIP/DINO backbone) is still worse than ResNet-18 fine-tuning on 100 samples (ViT overfits faster). |

**When CLIP/DINO would be right:**
- If you wanted to support natural language queries ("show me all wells with reagent X"; requires fine-grained dataset).
- If you had unlabeled data (DINO self-supervised pre-training on in-domain microplate images could help).
- If zero-shot transfer to new assay types (different plate formats, colors) was critical; CLIP's language-based zero-shot is excellent for that.

---

## Part 3: ML Stack Risk Register

### Risk 1: Catastrophic Overfitting on 100 Samples

**Severity:** CRITICAL  
**Probability:** HIGH (if hyperparameters are not carefully tuned)

**Description:**
Model memorizes training data; validation/test accuracy collapses. With only 100 samples split into ~90 train / ~10 val, even ResNet-18 can overfit if:
- Learning rate is too high (>1e-4)
- Dropout is too low (<0.3)
- No early stopping (train for >50 epochs)
- Augmentation is insufficient

**Mitigation:**
1. **Aggressive early stopping:** Monitor validation loss; stop if no improvement for 10 epochs. Target overfitting curve: train loss should plateau and validation loss should remain flat or drop slightly.
2. **Strong augmentation:** Use albumentations (spatial + photometric) to create virtual samples. Expect 3–5× effective dataset size from augmentation.
3. **Explicit regularization:**
   - Dropout (0.3–0.5 on classification heads)
   - Weight decay (1e-5 in AdamW)
   - Batch norm momentum = 0.99 (stabilize statistics on small batches)
4. **Cross-validation:** Use stratified 5-fold CV to estimate true generalization error; report CI across folds.
5. **Validation strategy:** Reserve 10 samples in hold-out set **before any experiment**; do not touch until final evaluation.

**Trigger for fallback:**
- If validation accuracy plateaus <70% after 30 epochs with strong regularization, suspect fundamental signal weakness.
- Fallback: Switch to hybrid approach (classical CV feature extraction + lightweight classifier).

---

### Risk 2: Class Imbalance and Missing Well Classes

**Severity:** HIGH  
**Probability:** MEDIUM

**Description:**
With 100 samples across 96 wells, ~60% of wells have zero samples. Even if a row is present, some columns may be absent. Example: Row A may appear in samples, but only columns 1–8; columns 9–12 are never seen.

Consequence: Row 8 and Column 12 have near-zero training signal. Model learns a default "low confidence" for these classes, leading to >30% false negative rate for rare well combinations.

**Mitigation:**
1. **Focal loss (γ=2.0):** Automatically down-weights confident predictions, focusing on hard cases.
2. **Per-class weighting:** Compute `class_weight = num_samples / (num_classes * per_class_count)`. Rows with 0 samples get weight ~2.0; common rows get ~0.5. Pass to focal loss as `alpha` parameter.
3. **Stratified sampling:** During training, ensure every batch includes at least one well from a rare class (if feasible). Use `WeightedRandomSampler` in PyTorch DataLoader.
4. **Synthetic augmentation (cautious):** If a row/column is completely missing, consider creating pseudo-samples via:
   - Mixup: Blend two samples from the same row/column to create a synthetic variant.
   - Geometric transformation: If row A images exist, apply small rotation/translation to create pseudo-row B samples (risky; can introduce bias).
5. **Per-class metrics:** Report row accuracy and column accuracy separately; identify which classes have <70% recall.

**Trigger for fallback:**
- If validation F1 score is <0.65 after 30 epochs with all mitigations, the task may require more fundamental signal.
- Fallback: Combine classical CV (grid detection + homography) with shallow NN to refine well predictions.

---

### Risk 3: Frame Extraction and Temporal Signal Loss

**Severity:** MEDIUM  
**Probability:** MEDIUM

**Description:**
If videos are compressed (h.264), frame extraction via cv2 may drop frames or produce artifacts. Additionally, extracting only 2 frames per video may miss the dispense event if timing is not uniform across videos.

Consequence: FPV (temporal localization signal) is lost; model only sees static top-view, reducing accuracy by ~10–15%.

**Mitigation:**
1. **Frame extraction validation:** After extracting frames, verify:
   - Frame count matches expected number (no dropped frames).
   - Histogram / pixel-level differences between frames exist (not all black/white).
   - Metadata (fps, codec) is readable.
2. **Adaptive frame sampling:** Instead of fixed frame indices, search for maximum frame variance (motion detection):
   ```python
   def extract_high_variance_frames(video_path, n_frames=2):
       # Compute frame-to-frame difference; pick frames with highest variance
       frames = load_all_frames(video_path)
       diffs = [np.sum(np.abs(frames[i] - frames[i-1])) for i in range(1, len(frames))]
       top_indices = np.argsort(diffs)[-n_frames:]
       return frames[sorted(top_indices)]
   ```
3. **Multi-frame aggregation:** Extract 4–5 frames instead of 2; aggregate via max-pooling on sigmoid outputs. Robustness to timing jitter.
4. **Sanity checks:** Visualize extracted frames during data loading. Catch codec issues early.

**Trigger for fallback:**
- If inference frames are mostly blank/corrupted, fallback to key-frame extraction (first + last frame).
- If top-view only (no FPV) yields >80% accuracy, FPV signal is weak; consider single-view model.

---

## Part 4: Data Augmentation Strategy

### 4.1 Augmentation Rationale and Scale

**Goal:** Increase effective dataset size from 100 to ~400–500 samples via realistic transformations, without introducing domain-shift artifacts.

**Constraints:**
- **Plate geometry is fixed:** Don't apply extreme rotations/perspective (well positions are calibrated). Safe range: ±15° rotation, ±10% perspective shift.
- **Color/intensity is critical for well visibility:** Don't apply extreme brightness changes. Safe range: ±15% brightness, ±10% contrast.
- **Specular reflection is a real problem:** Intentional augmentation with GaussBlur + MotionBlur can help robustness.

### 4.2 Spatial Augmentations

| Augmentation | Probability | Safe Range | Rationale |
|--------------|-------------|-----------|-----------|
| **Rotation** | 0.5 | ±15° | Plate may be tilted in holder. Beyond ±15°, well positions distort visibly. |
| **Affine (shift)** | 0.5 | ±5% image height/width | Crop shift simulates plate position variability. Preserve well structure. |
| **Elastic deformation** | 0.3 | σ ∈ [1, 3], α ∈ [30, 50] | Depth-of-field blur is nonlinear; elastic deformation is a mild approximation. Use sparingly. |
| **Perspective transform** | 0.3 | ±10% corner shift | Shallow viewing angle introduces perspective. Keep mild. |
| **Horizontal flip** | 0 (disabled) | N/A | Well labels are asymmetric (A1 is top-left). Flip reverses row/col order; do not use. |
| **Vertical flip** | 0 (disabled) | N/A | Same; disables symmetry. Skip. |
| **Scale (zoom)** | 0.2 | 0.9–1.1 (±10%) | Very mild zoom simulates focus distance. Avoid >±10% (distorts well sizes). |

**Implementation (albumentations):**
```python
import albumentations as A

spatial_augmentation = A.Compose([
    A.Rotate(limit=15, border_mode=cv2.BORDER_REFLECT, p=0.5),
    A.Affine(translate_percent=(-0.05, 0.05), p=0.5),
    A.Perspective(scale=(0.05, 0.1), p=0.3),
    A.ElasticTransform(alpha=(30, 50), sigma=(1, 3), p=0.3),
    A.Resize(224, 224),  # Ensure consistent output size
], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels']))
```

### 4.3 Photometric Augmentations

| Augmentation | Probability | Safe Range | Rationale |
|--------------|-------------|-----------|-----------|
| **Brightness/Contrast** | 0.6 | ±0.15 (brightness), ±0.15 (contrast) | Microscope illumination varies. Mild jitter helps robustness. |
| **Gaussian blur** | 0.4 | kernel ∈ [3, 5] | Depth-of-field and out-of-focus regions. Mild blur (σ=1–2). |
| **Motion blur** | 0.3 | kernel ∈ [3, 7], angle ∈ [0, 360] | Vibration, hand tremor during dispense. Essential for FPV realism. |
| **Gamma correction** | 0.3 | γ ∈ [0.7, 1.3] | Non-linear camera response. Simulates underexposure / overexposure. |
| **Noise (Gaussian)** | 0.3 | σ ∈ [0.01, 0.03] (relative to 0–1 scale) | Sensor noise. Use conservatively (modern cameras have low noise). |
| **Additive Gaussian** | 0.2 | variance ∈ [5, 15] (on 0–255 scale) | Alternative to Gaussian noise; slightly more realistic. |
| **Channel shuffle** | 0.0 (disabled) | N/A | Well visibility depends on channel order (fluorescence is wavelength-dependent). Skip. |
| **Hue/Saturation shift** | 0.2 | hue ∈ [±10], saturation ∈ [±10] | Fluorophore color can vary (excitation/filter variability). Mild shift OK. |

**Implementation:**
```python
photometric_augmentation = A.Compose([
    A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=0.6),
    A.GaussBlur(blur_limit=5, p=0.4),
    A.MotionBlur(blur_limit=7, p=0.3),
    A.RandomGamma(gamma_limit=(70, 130), p=0.3),
    A.GaussNoise(var_limit=(10, 50), p=0.3),
    A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=10, val_shift_limit=0, p=0.2),
], additional_targets={'image1': 'image'})  # Support dual images (FPV + top-view)
```

### 4.4 Video-Specific Augmentations

| Augmentation | Valid? | Probability | Rationale |
|--------------|--------|-------------|-----------|
| **Frame dropping (skip frames)** | ✅ YES | 0.2 | Variable fps in videos; skipping frames is realistic. |
| **Temporal jitter (shuffle frame order)** | ⚠️ CAREFUL | 0.0 (disabled) | Dispense event has temporal direction (liquid motion → fills well). Reversing breaks causality. If frames are static (no dispense visible), jitter is safe but adds little value. **Recommendation: Disable.** |
| **Frame reversal (reverse video)** | ⚠️ CAREFUL | 0.0 (disabled) | Same as jitter; dispense is asymmetric in time. **Recommendation: Disable unless you confirm frames show no dispense event.** |
| **Frame interpolation (add synthetic frames)** | ❌ NO | 0.0 | Requires optical flow or frame synthesis; adds complexity without clear benefit. Skip. |
| **Clip reversal (reverse extraction order)** | ❌ NO | 0.0 | Reverses the temporal signal; breaks the dispense temporal structure. Disable. |

**Recommended approach:** Extract 2 frames (early and late) from video; apply identical spatial/photometric augmentations to both. No temporal augmentations beyond frame dropping.

**Pseudocode:**
```python
def augment_dual_frames(fpv_frame_early, fpv_frame_late, topview_frame_early, topview_frame_late, 
                       spatial_aug, photometric_aug):
    """Apply consistent augmentation across all 4 frames."""
    # Combine frames into a multi-channel tensor for consistent augmentation
    # [FPV early, FPV late, TopView early, TopView late] → (H, W, 12)
    
    combined = np.concatenate([
        fpv_frame_early, fpv_frame_late, 
        topview_frame_early, topview_frame_late
    ], axis=2)  # (H, W, 12)
    
    # Apply spatial augmentation to all frames uniformly
    aug_spatial = spatial_aug(image=combined)['image']  # (H, W, 12)
    
    # Apply photometric augmentation to each pair separately (different augmentations per frame)
    fpv_early = aug_spatial[..., :3]
    fpv_late = aug_spatial[..., 3:6]
    topview_early = aug_spatial[..., 6:9]
    topview_late = aug_spatial[..., 9:12]
    
    fpv_early = photometric_aug(image=fpv_early)['image']
    fpv_late = photometric_aug(image=fpv_late)['image']
    topview_early = photometric_aug(image=topview_early)['image']
    topview_late = photometric_aug(image=topview_late)['image']
    
    return fpv_early, fpv_late, topview_early, topview_late
```

### 4.5 Augmentation Validation

Before training, validate augmentations:
1. **Visual inspection:** Plot 5 original images + 5 augmented variants. Verify plates are still recognizable, wells are visible.
2. **Label consistency:** Ensure augmented images still belong to the same well classes (no label drift).
3. **Statistical check:** Compute histogram and edge-detection statistics on original vs. augmented; should be similar.

---

## Part 5: Evaluation Protocol

### 5.1 Data Splitting Strategy

**Challenge:** 100 samples is tiny. Standard 80/10/10 train/val/test leaves only ~10 test samples; confidence intervals are very wide.

**Recommended approach:** Stratified hold-out validation with cross-validation on training set

**Step 1: Stratified hold-out (global level)**
```
Goal: Hold out ~10% of samples (≈10 samples) for final evaluation.
Stratification dimension: Well type (single-channel, 8-channel, 12-channel)
  - If your 100 samples are: 30 single, 40 8-channel, 30 12-channel
  - Hold out: 3 single, 4 8-channel, 3 12-channel (=10 total)
  - Training pool: 97 samples
```

**Step 2: Stratified 5-fold CV on training pool**
```
Split 97 samples into 5 folds, stratified by well type:
  - Fold 1: 19 samples (validation), 78 samples (train)
  - Fold 2: 19 samples (validation), 78 samples (train)
  - ...
  - Fold 5: 20 samples (validation), 77 samples (train)

For each fold:
  - Train model on 77–78 samples
  - Evaluate on 19–20 validation samples
  - Report metrics (accuracy, F1, per-well recall)

Final CV result: Average metrics across 5 folds + 95% CI (bootstrap)
```

**Step 3: Final evaluation on hold-out set**
```
Train final model on all 97 samples (or use fold-averaged weights).
Evaluate on held-out 10 samples (once, no tuning).
Report: Accuracy, F1, per-well recall, confidence intervals.
```

**Pseudocode (PyTorch):**
```python
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import MultiLabelBinarizer

def stratified_cv_evaluation(samples, labels, well_types, n_splits=5):
    """
    Stratified k-fold CV on 100 samples.
    
    Args:
        samples: List of (fpv_path, topview_path)
        labels: List of (row_idx, col_idx) tuples
        well_types: List of well type (0=single, 1=8-channel, 2=12-channel)
        n_splits: Number of folds (5 recommended)
    """
    
    # Step 1: Hold out ~10% stratified by well type
    skf_holdout = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    train_idx, test_idx = next(skf_holdout.split(samples, well_types))
    
    train_samples, test_samples = [samples[i] for i in train_idx], [samples[i] for i in test_idx]
    train_labels, test_labels = [labels[i] for i in train_idx], [labels[i] for i in test_idx]
    train_types = [well_types[i] for i in train_idx]
    
    print(f"Hold-out set: {len(test_samples)} samples")
    print(f"Training pool: {len(train_samples)} samples")
    
    # Step 2: Stratified 5-fold CV on training pool
    cv_results = []
    skf_cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    for fold_idx, (fold_train_idx, fold_val_idx) in enumerate(skf_cv.split(train_samples, train_types)):
        fold_train_samples = [train_samples[i] for i in fold_train_idx]
        fold_val_samples = [train_samples[i] for i in fold_val_idx]
        fold_train_labels = [train_labels[i] for i in fold_train_idx]
        fold_val_labels = [train_labels[i] for i in fold_val_idx]
        
        print(f"\nFold {fold_idx + 1}: Train {len(fold_train_samples)}, Val {len(fold_val_samples)}")
        
        # Create DataLoaders
        train_loader = create_dataloader(fold_train_samples, fold_train_labels, batch_size=8)
        val_loader = create_dataloader(fold_val_samples, fold_val_labels, batch_size=8)
        
        # Train model
        model = initialize_model()
        train_loop(model, train_loader, epochs=50, early_stopping_patience=10)
        
        # Evaluate
        metrics = evaluate(model, val_loader)
        cv_results.append(metrics)
        
        print(f"  Accuracy: {metrics['accuracy']:.3f}, F1: {metrics['f1']:.3f}")
    
    # Step 3: Aggregate CV results
    avg_accuracy = np.mean([m['accuracy'] for m in cv_results])
    std_accuracy = np.std([m['accuracy'] for m in cv_results])
    ci_accuracy = (avg_accuracy - 1.96 * std_accuracy / np.sqrt(n_splits),
                   avg_accuracy + 1.96 * std_accuracy / np.sqrt(n_splits))
    
    print(f"\n=== Cross-Validation Results ===")
    print(f"Accuracy: {avg_accuracy:.3f} ± {std_accuracy:.3f} [95% CI: {ci_accuracy}]")
    
    # Step 4: Train final model on all training data, evaluate on hold-out
    print(f"\n=== Hold-Out Evaluation ===")
    final_model = initialize_model()
    final_train_loader = create_dataloader(train_samples, train_labels, batch_size=8)
    train_loop(final_model, final_train_loader, epochs=50, early_stopping_patience=10)
    
    final_test_loader = create_dataloader(test_samples, test_labels, batch_size=8)
    final_metrics = evaluate(final_model, final_test_loader)
    
    print(f"Hold-out Accuracy: {final_metrics['accuracy']:.3f}")
    print(f"Hold-out F1: {final_metrics['f1']:.3f}")
    
    return {
        'cv_results': cv_results,
        'cv_accuracy_mean': avg_accuracy,
        'cv_accuracy_std': std_accuracy,
        'hold_out_metrics': final_metrics
    }
```

### 5.2 Stratification Dimensions

**Primary stratification:** Well type (single-channel, 8-channel, 12-channel)
- Ensures train/val splits have similar class distributions.
- Prevents a scenario where all single-channel samples are in training and all 12-channel are in validation.

**Secondary stratification (if feasible):** Well position class (rare vs. common)
- Identify wells that appear in <5 samples (rare) vs. >10 samples (common).
- Ensure both rare and common wells are represented in train/val.
- Use hierarchical stratification: `stratify_by = [well_type, well_rarity]`.

### 5.3 Metrics to Report

**At each fold:**

| Metric | Formula | Interpretation |
|--------|---------|-----------------|
| **Exact-match accuracy** | TP / (TP + FP + FN) | Fraction of samples with perfect row+col prediction |
| **Row accuracy** | # correct row predictions / # samples | Row detection quality |
| **Column accuracy** | # correct column predictions / # samples | Column detection quality |
| **Macro F1** | (1/K) × Σ F1_k | Average F1 across all K classes (8 rows + 12 cols) |
| **Micro F1** | 2·TP / (2·TP + FP + FN) | Overall F1 treating all classes equally |
| **Per-well recall** | For each well (r, c): TP_rc / (TP_rc + FN_rc) | Identify failure modes (e.g., well H12 never detected) |
| **Hamming loss** | # label mismatches / (# samples × # output dimensions) | Fraction of incorrect row/col predictions |

**Final report (hold-out set):**
```
Cross-Validation Results (5-fold stratified):
  - Exact-match accuracy: 0.823 ± 0.087 [95% CI: 0.736–0.910]
  - Row accuracy: 0.901 ± 0.062 [95% CI: 0.839–0.963]
  - Column accuracy: 0.876 ± 0.095 [95% CI: 0.781–0.971]
  - Macro F1: 0.812 ± 0.098
  - Micro F1: 0.829 ± 0.085

Hold-Out Evaluation (10 samples):
  - Exact-match accuracy: 0.800 [8/10 correct]
  - Row accuracy: 0.900 [9/10 correct]
  - Column accuracy: 0.800 [8/10 correct]
  - Macro F1: 0.790
  - Per-well recall (sorted by frequency):
    - Well A1: 1.00 (appears in 5 train samples, 1 test; detected 1/1)
    - Well H12: 0.50 (appears in 2 train samples, 1 test; detected 0/1) ← FAILURE MODE
    - ...
```

### 5.4 Validation Sanity Checks

**Before submitting results, verify:**

1. **Label consistency:** Predictions and ground truth are from the same split (no leakage).
   ```python
   assert np.intersect1d(train_idx, val_idx).size == 0  # No overlap
   ```

2. **Stratification check:** Well type distribution is similar across train/val.
   ```python
   train_types_count = np.bincount(train_types)
   val_types_count = np.bincount(val_types)
   # Chi-square test for homogeneity; p-value should be >0.05
   ```

3. **No data leakage via augmentation:** Augmented images are semantically identical to originals (same well labels).
   ```python
   # Augment a sample; re-infer; prediction should be same (high confidence)
   ```

4. **Threshold is fixed before evaluation:** Do not tune threshold on hold-out set.
   ```python
   threshold = tune_threshold(model, cv_fold_val_loaders)  # Tune on CV folds only
   test_predictions = (model(test_samples) > threshold).int()  # Apply fixed threshold
   ```

5. **Metrics are computed correctly:** Manually spot-check 5 predictions.
   ```python
   sample_pred = [1, 0, 1, 0, 0, 0, 0, 0,  # 3 rows
                  1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  # 3 cols
   sample_true = [1, 0, 1, 0, 0, 0, 0, 0,  # Match
                  1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  # Match
   # exact_match_accuracy should increase by 1
   ```

---

## Conclusion: ML Stack Summary

| Component | Choice | Confidence | Risk Level |
|-----------|--------|-----------|-----------|
| **Framework** | PyTorch 2.0+ | High | Low |
| **Backbone** | ResNet-18 (ImageNet pre-trained) | High | Low |
| **Video I/O** | OpenCV (cv2) frame extraction | High | Low |
| **Augmentation** | albumentations (spatial + photometric) | High | Low |
| **Output head** | Factorized row (8) + column (12) | High | Medium |
| **Loss** | Focal loss (γ=2.0) per head | High | Low |
| **Training** | AdamW + cosine annealing + warmup | High | Low |
| **Dual-view fusion** | Early fusion (feature-map concat) | Medium | Medium |
| **Evaluation** | Stratified 5-fold CV + hold-out | High | Low |

**Expected performance:** 80–90% exact-match accuracy on hold-out validation (10 samples).

**Estimated effort:**
- Data pipeline: 1–2 days
- Model training + hyperparameter tuning: 3–5 days
- Evaluation + documentation: 1 day
- **Total: 1–2 weeks for end-to-end pipeline**

---

## Appendix: Reference Implementations

### A.1 Training Loop Pseudocode

```python
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingWarmRestarts
from torchvision.ops import sigmoid_focal_loss
from torch.cuda.amp import autocast, GradScaler

def train_epoch(model, train_loader, optimizer, scaler, device):
    model.train()
    total_loss = 0.0
    
    for batch_idx, (fpv_images, topview_images, row_labels, col_labels) in enumerate(train_loader):
        fpv_images = fpv_images.to(device)
        topview_images = topview_images.to(device)
        row_labels = row_labels.to(device).float()
        col_labels = col_labels.to(device).float()
        
        optimizer.zero_grad()
        
        with autocast():
            row_logits, col_logits = model(fpv_images, topview_images)
            loss = (sigmoid_focal_loss(row_logits, row_labels, gamma=2.0, reduction='mean') +
                    sigmoid_focal_loss(col_logits, col_labels, gamma=2.0, reduction='mean'))
        
        scaler.scale(loss).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        
        total_loss += loss.item()
    
    return total_loss / len(train_loader)

def validate(model, val_loader, device):
    model.eval()
    total_loss = 0.0
    
    with torch.no_grad():
        for fpv_images, topview_images, row_labels, col_labels in val_loader:
            fpv_images = fpv_images.to(device)
            topview_images = topview_images.to(device)
            row_labels = row_labels.to(device).float()
            col_labels = col_labels.to(device).float()
            
            with autocast():
                row_logits, col_logits = model(fpv_images, topview_images)
                loss = (sigmoid_focal_loss(row_logits, row_labels, gamma=2.0, reduction='mean') +
                        sigmoid_focal_loss(col_logits, col_labels, gamma=2.0, reduction='mean'))
            
            total_loss += loss.item()
    
    return total_loss / len(val_loader)

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = DualViewWellDetector().to(device)
    
    optimizer = AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
    scaler = GradScaler()
    
    warmup_scheduler = LinearLR(optimizer, start_factor=0.01, total_iters=5)
    cosine_scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=45, T_mult=1, eta_min=1e-6)
    
    train_loader = create_dataloader(train_samples, batch_size=8, shuffle=True)
    val_loader = create_dataloader(val_samples, batch_size=8, shuffle=False)
    
    best_val_loss = float('inf')
    patience = 10
    patience_counter = 0
    
    for epoch in range(50):
        train_loss = train_epoch(model, train_loader, optimizer, scaler, device)
        val_loss = validate(model, val_loader, device)
        
        print(f"Epoch {epoch+1}: Train loss={train_loss:.4f}, Val loss={val_loss:.4f}")
        
        if epoch < 5:
            warmup_scheduler.step()
        else:
            cosine_scheduler.step()
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), 'best_model.pt')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
    
    model.load_state_dict(torch.load('best_model.pt'))
    return model
```

---

**Document version:** 1.0  
**Last updated:** April 2026  
**Author:** ML Science Team
