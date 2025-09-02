import os
import sys
import time
import json
import torch
import random
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
from sklearn.metrics import roc_auc_score, accuracy_score
from fairlearn.metrics import demographic_parity_difference, demographic_parity_ratio, equalized_odds_difference, equalized_odds_ratio
from models import ClsResNet
from datasets import RNFLT_Dataset_Pred_Tds

# ----------------------------
# Configuration
# ----------------------------
BASE_DIR = Path(__file__).parent
TRAIN_PATH = BASE_DIR / 'dataset' / 'Training'
VAL_PATH   = BASE_DIR / 'dataset' / 'Validation'
MODALITY   = 'combined'
BATCH_SIZE = 16
NUM_EPOCHS = 5 # for demonstration; use 50 or more in practice
LR         = 1e-4
NUM_CLASSES = 1  # regression
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ----------------------------
# Dataset and Dataloader
# ----------------------------
train_dataset = RNFLT_Dataset_Pred_Tds(TRAIN_PATH, subset='train', modality_type=MODALITY)
val_dataset   = RNFLT_Dataset_Pred_Tds(VAL_PATH, subset='val', modality_type=MODALITY)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
val_loader   = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

# ----------------------------
# Model, Loss, Optimizer
# ----------------------------
in_channels = 2 if MODALITY == 'combined' else 1
model = ClsResNet(pretrained=False, num_classes=NUM_CLASSES, backbone='resnet18', in_channels=in_channels).to(DEVICE)
criterion = nn.MSELoss()  # regression
optimizer = optim.Adam(model.parameters(), lr=LR)

# ----------------------------
# Utility functions
# ----------------------------
def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

def compute_metrics(preds, labels, attrs=None):
    preds_np = preds.detach().cpu().numpy()
    labels_np = labels.detach().cpu().numpy()
    
    # simple thresholding for regression to binary
    binary_preds = (preds_np >= np.median(preds_np)).astype(int)
    binary_labels = (labels_np >= np.median(labels_np)).astype(int)
    
    acc = accuracy_score(binary_labels, binary_preds)
    try:
        auc = roc_auc_score(binary_labels, preds_np)
    except ValueError:
        auc = float('nan')

    dpd = demographic_parity_difference(binary_labels, binary_preds, sensitive_features=attrs) if attrs is not None else np.nan
    dpr = demographic_parity_ratio(binary_labels, binary_preds, sensitive_features=attrs) if attrs is not None else np.nan
    eod = equalized_odds_difference(binary_labels, binary_preds, sensitive_features=attrs) if attrs is not None else np.nan
    eor = equalized_odds_ratio(binary_labels, binary_preds, sensitive_features=attrs) if attrs is not None else np.nan

    return acc, auc, dpd, dpr, eod, eor

def train_one_epoch(model, loader, optimizer, criterion):
    model.train()
    running_loss = 0.0
    all_preds, all_labels, all_attrs = [], [], []
    
    for X, y in loader:
        X, y = X.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(X)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * X.size(0)
        all_preds.append(outputs)
        all_labels.append(y)
        # dummy attributes, replace with actual if available
        all_attrs.append(torch.zeros_like(y))  
    
    avg_loss = running_loss / len(loader.dataset)
    all_preds = torch.cat(all_preds, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    all_attrs = torch.cat(all_attrs, dim=0).cpu().numpy()
    
    acc, auc, dpd, dpr, eod, eor = compute_metrics(all_preds, all_labels, all_attrs)
    
    return avg_loss, acc, auc, dpd, dpr, eod, eor, all_preds, all_labels, all_attrs

def validate(model, loader, criterion):
    model.eval()
    running_loss = 0.0
    all_preds, all_labels, all_attrs = [], [], []
    
    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(DEVICE), y.to(DEVICE)
            outputs = model(X)
            loss = criterion(outputs, y)
            running_loss += loss.item() * X.size(0)
            all_preds.append(outputs)
            all_labels.append(y)
            all_attrs.append(torch.zeros_like(y))  # dummy attributes
    
    avg_loss = running_loss / len(loader.dataset)
    all_preds = torch.cat(all_preds, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    all_attrs = torch.cat(all_attrs, dim=0).cpu().numpy()
    
    acc, auc, dpd, dpr, eod, eor = compute_metrics(all_preds, all_labels, all_attrs)
    
    return avg_loss, acc, auc, dpd, dpr, eod, eor, all_preds, all_labels, all_attrs

# ----------------------------
# Training Loop
# ----------------------------
best_auc = 0.0
for epoch in range(NUM_EPOCHS):
    train_loss, train_acc, train_auc, train_dpd, train_dpr, train_eod, train_eor, _, _, _ = train_one_epoch(model, train_loader, optimizer, criterion)
    val_loss, val_acc, val_auc, val_dpd, val_dpr, val_eod, val_eor, val_preds, val_labels, val_attrs = validate(model, val_loader, criterion)
    
    print(f"Epoch {epoch+1}/{NUM_EPOCHS}")
    print(f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f}, AUC: {train_auc:.4f}")
    print(f"Val   Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, AUC: {val_auc:.4f}")
    print(f"Fairness - DPD: {val_dpd:.4f}, DPR: {val_dpr:.4f}, EOD: {val_eod:.4f}, EOR: {val_eor:.4f}")
    
    if val_auc > best_auc:
        best_auc = val_auc
        torch.save(model.state_dict(), f'best_baseline_resnet_{MODALITY}.pth')
        print(f"Best model updated at epoch {epoch+1} with AUC: {best_auc:.4f}")

# Save final model
torch.save(model.state_dict(), f'final_baseline_resnet_{MODALITY}.pth')
print(f"Final model saved as final_baseline_resnet_{MODALITY}.pth")
