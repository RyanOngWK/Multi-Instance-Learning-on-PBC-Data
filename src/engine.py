import torch
import numpy as np
import os
import sys
from .utils import compute_multiclass_metrics, save_attention_heatmap

def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    y_true, y_pred = [], []
    for batch in loader:
        patches = batch["patches"].to(device, non_blocking=True)
        labels  = batch["labels"].to(device, non_blocking=True)
        masks   = batch["masks"].to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        logits, attn = model(patches, mask=masks)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * labels.size(0)
        preds = torch.argmax(logits, dim=1).detach().cpu().numpy()
        # preds = (torch.sigmoid(logits) > 0.5).detach().cpu().numpy().astype(int)
        # labels_np = labels.detach().cpu().numpy().astype(int)
        labels_np = labels.detach().cpu().numpy()
        y_true.extend(labels_np.tolist())
        y_pred.extend(preds.tolist())
    metrics = compute_multiclass_metrics(y_true, y_pred)
    metrics.loss = running_loss / max(len(loader.dataset), 1)
    return metrics

@torch.no_grad()
def evaluate(model, loader, criterion, device, heatmap_dir=None, tile_size=224):
    model.eval()
    running_loss = 0.0
    y_true, y_pred = [], []
    if heatmap_dir is not None:
        os.makedirs(heatmap_dir, exist_ok=True)
    saved = set()
    for batch in loader:
        patches = batch["patches"].to(device, non_blocking=True)
        labels  = batch["labels"].to(device, non_blocking=True)
        masks   = batch["masks"].to(device, non_blocking=True)
        logits, attn = model(patches, mask=masks)   
        loss = criterion(logits, labels)
        running_loss += loss.item() * labels.size(0)
        preds = torch.argmax(logits, dim=1).cpu().numpy()
        # preds = (torch.sigmoid(logits) > 0.5).detach().cpu().numpy().astype(int)
        # labels_np = labels.detach().cpu().numpy().astype(int)
        labels_np = labels.cpu().numpy()
        y_true.extend(labels_np.tolist())
        y_pred.extend(preds.tolist())
        if heatmap_dir is not None:
            attn_np = attn.cpu().numpy()
            for i in range(len(batch["image_ids"])):
                image_id = batch["image_ids"][i]
                if image_id in saved:
                    continue
                out_path = os.path.join(heatmap_dir, f"{image_id}_attention_heatmap.png")
                try:
                    save_attention_heatmap(
                        image_path=batch["image_paths"][i],
                        coords=batch["coords"][i],
                        attn=attn_np[i],
                        tile_size=tile_size,
                        out_path=out_path,
                    )
                except Exception as e:
                    print(f"[WARN] Failed to save heatmap for {image_id}: {e}", file=sys.stderr)
                saved.add(image_id)
    metrics = compute_multiclass_metrics(y_true, y_pred)
    metrics.loss = running_loss / max(len(loader.dataset), 1)
    return metrics, y_true, y_pred   

