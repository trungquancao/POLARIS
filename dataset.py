import os
import cv2
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

class IsicDataset(Dataset):
    def __init__(self, img_dir, mask_dir, img_size=224):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.img_size = img_size
        
        self.images = sorted([f for f in os.listdir(img_dir) if f.endswith('.jpg') or f.endswith('.png')])
        
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        base_name = img_name.split('.')[0]
        mask_name = f"{base_name}_segmentation.png"
        
        if "_aug_" in img_name:
            parts = img_name.replace(".jpg", "").split("_aug_")
            mask_name = f"{parts[0]}_segmentation_aug_{parts[1]}.png"

        img_path = os.path.join(self.img_dir, img_name)
        mask_path = os.path.join(self.mask_dir, mask_name)

        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (self.img_size, self.img_size))

        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(mask, (self.img_size, self.img_size), interpolation=cv2.INTER_NEAREST)

        image_tensor = self.transform(image) 
        
        mask_tensor = torch.from_numpy(mask).float() / 255.0
        mask_tensor = mask_tensor.unsqueeze(0)

        return image_tensor, mask_tensor


class HamDataset(Dataset):
    def __init__(self, csv_file, img_dir, img_size=224):
        self.img_dir = img_dir
        self.img_size = img_size
        self.df = pd.read_csv(csv_file)
        
        self.label_map = {'nv': 0, 'mel': 1, 'bkl': 2, 'bcc': 3, 'akiec': 4, 'vasc': 5, 'df': 6}
        
        self.sex_map = {'male': 0.0, 'female': 1.0, 'unknown': 0.5}
        
        locations = self.df['localization'].unique().tolist()
        self.loc_map = {loc: i/len(locations) for i, loc in enumerate(locations)}

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        img_name = f"{row['image_id']}.jpg"
        img_path = os.path.join(self.img_dir, img_name)
        
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (self.img_size, self.img_size))
        image_tensor = self.transform(image)

        age = row['age'] if not pd.isna(row['age']) else 50.0 # Ca nào mất thông tin cho mặc định 50 tuổi
        age_norm = age / 100.0

        sex_norm = self.sex_map.get(row['sex'], 0.5)
        
        loc_norm = self.loc_map.get(row['localization'], 0.0)
        
        growth_rate = 0.0 
        
        metadata_tensor = torch.tensor([age_norm, sex_norm, loc_norm, growth_rate], dtype=torch.float32)

        label_str = row['dx']
        label_tensor = torch.tensor(self.label_map[label_str], dtype=torch.long)

        return image_tensor, metadata_tensor, label_tensor

