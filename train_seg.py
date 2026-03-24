import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
import torch.nn.functional as F

from dataset import IsicDataset
from models import PolarisSeg

class DiceFocalLoss(nn.Module):
    def __init__(self, dice_weight=0.5, focal_weight=0.5, gamma=2.0, alpha=0.25, smooth=1e-6):
        super(DiceFocalLoss, self).__init__()
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight
        self.gamma = gamma
        self.alpha = alpha
        self.smooth = smooth

    def forward(self, logits, targets):
        bce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        focal_loss = focal_loss.mean()

        probs = torch.sigmoid(logits)
        intersection = (probs * targets).sum(dim=(2, 3))
        union = probs.sum(dim=(2, 3)) + targets.sum(dim=(2, 3))
        dice_score = (2.0 * intersection + self.smooth) / (union + self.smooth)
        dice_loss = 1.0 - dice_score.mean()

        return self.dice_weight * dice_loss + self.focal_weight * focal_loss

def calculate_dice(logits, targets, threshold=0.5, smooth=1e-6):
    probs = torch.sigmoid(logits)
    preds = (probs > threshold).float()
    intersection = (preds * targets).sum(dim=(2, 3))
    union = preds.sum(dim=(2, 3)) + targets.sum(dim=(2, 3))
    dice = (2.0 * intersection + smooth) / (union + smooth)
    return dice.mean().item()

def train_segmentation():
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    EPOCHS = 100         
    BATCH_SIZE = 16      
    LR = 3e-4           
    PATIENCE_LR = 5    
    PATIENCE_STOP = 15

    TRAIN_IMG = "ISIC_Split_Dataset/images/train"
    TRAIN_MASK = "ISIC_Split_Dataset/masks/train"
    VAL_IMG = "ISIC_Split_Dataset/images/val"
    VAL_MASK = "ISIC_Split_Dataset/masks/val"

    train_dataset = IsicDataset(TRAIN_IMG, TRAIN_MASK)
    val_dataset = IsicDataset(VAL_IMG, VAL_MASK)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    model = PolarisSeg().to(device)
    
    criterion = DiceFocalLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=PATIENCE_LR)

    best_val_loss = float('inf')
    epochs_no_improve = 0
    save_dir = "weights"
    os.makedirs(save_dir, exist_ok=True)

    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0
        train_dice = 0.0
        
        pbar = tqdm(train_loader, desc="Training Seg")
        for images, masks in pbar:
            images, masks = images.to(device), masks.to(device)
            
            optimizer.zero_grad()
            logits = model(images)
            
            loss = criterion(logits, masks)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * images.size(0)
            train_dice += calculate_dice(logits, masks) * images.size(0)
            
            pbar.set_postfix({'Loss': f"{loss.item():.4f}"})
            
        epoch_train_loss = train_loss / len(train_dataset)
        epoch_train_dice = train_dice / len(train_dataset)

        model.eval()
        val_loss = 0.0
        val_dice = 0.0
        
        with torch.no_grad():
            for images, masks in tqdm(val_loader, desc="Validating"):
                images, masks = images.to(device), masks.to(device)
                
                logits = model(images)
                loss = criterion(logits, masks)
                
                val_loss += loss.item() * images.size(0)
                val_dice += calculate_dice(logits, masks) * images.size(0)
                
        epoch_val_loss = val_loss / len(val_dataset)
        epoch_val_dice = val_dice / len(val_dataset)

        scheduler.step(epoch_val_loss)

        print(f"\n\EPOCH {epoch+1}:")
        print(f"   ┣━ Train Loss: {epoch_train_loss:.4f} | Train Dice (Acc): {epoch_train_dice*100:.2f}%")
        print(f"   ┗━ Val Loss  : {epoch_val_loss:.4f} | Val Dice (Acc)  : {epoch_val_dice*100:.2f}%")

        # --- CHECKPOINT VÀ EARLY STOPPING ---
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            epochs_no_improve = 0
            save_path = os.path.join(save_dir, 'polaris_seg_best.pt')
            torch.save(model.state_dict(), save_path)
        else:
            epochs_no_improve += 1
            
            if epochs_no_improve >= PATIENCE_STOP:
                break

if __name__ == "__main__":
    train_segmentation()