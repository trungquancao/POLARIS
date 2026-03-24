import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
import torch.nn.functional as F

from dataset import HamDataset
from models import PolarisMultimodal

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()

def train_classification():
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    EPOCHS = 30
    BATCH_SIZE = 32
    LR = 1e-4 
    
    TRAIN_CSV = "HAM10000_Split_Dataset/train_meta_augmented.csv"
    TRAIN_IMG = "HAM10000_Split_Dataset/images/train"
    VAL_CSV = "HAM10000_Split_Dataset/val_meta.csv"
    VAL_IMG = "HAM10000_Split_Dataset/images/val"

    train_dataset = HamDataset(TRAIN_CSV, TRAIN_IMG)
    val_dataset = HamDataset(VAL_CSV, VAL_IMG)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    model = PolarisMultimodal(num_classes=7, meta_features=4).to(device)
    
    criterion = FocalLoss(gamma=2.0)
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS)

    best_val_loss = float('inf')
    save_dir = "weights"
    os.makedirs(save_dir, exist_ok=True)

    for epoch in range(EPOCHS):
        print(f"\n{'='*30}\n EPOCH {epoch+1}/{EPOCHS}\n{'='*30}")
        
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        pbar = tqdm(train_loader, desc="Training")
        for images, metadata, labels in pbar:
            images, metadata, labels = images.to(device), metadata.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images, metadata)
            
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
            
            pbar.set_postfix({'Loss': f"{loss.item():.4f}"})
            
        scheduler.step() 
        
        epoch_train_loss = train_loss / train_total
        epoch_train_acc = (train_correct / train_total) * 100

        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad(): 
            for images, metadata, labels in tqdm(val_loader, desc="Validating"):
                images, metadata, labels = images.to(device), metadata.to(device), labels.to(device)
                
                outputs = model(images, metadata)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item() * images.size(0)
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
                
        epoch_val_loss = val_loss / val_total
        epoch_val_acc = (val_correct / val_total) * 100

        print(f"\n KẾT QUẢ EPOCH {epoch+1}:")
        print(f"   ┣━ Train Loss: {epoch_train_loss:.4f} | Train Acc: {epoch_train_acc:.2f}%")
        print(f"   ┗━ Val Loss  : {epoch_val_loss:.4f} | Val Acc  : {epoch_val_acc:.2f}%")


        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            save_path = os.path.join(save_dir, 'polaris_cls_best.pt')
            torch.save(model.state_dict(), save_path)
            print(f"Save model: {save_path}")

if __name__ == "__main__":
    train_classification()