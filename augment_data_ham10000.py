import os
import cv2
import pandas as pd
import numpy as np
import albumentations as A
from tqdm import tqdm

def capped_augmentation_ham10000(train_dir, csv_path, output_csv_path, target_count=6000, max_multiplier=15):
    df = pd.read_csv(csv_path)
    class_counts = df['dx'].value_counts()
    
    multiplier_dict = {}
    
    for dx, count in class_counts.items():
        if count >= target_count:
            multiplier_dict[dx] = 0
        else:
            needed = target_count - count
            mult = int(np.ceil(needed / count))
            
            final_mult = min(mult, max_multiplier)
            multiplier_dict[dx] = final_mult
            
            expected_total = count + (count * final_mult)

    transform = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.Affine(scale=(0.9, 1.1), translate_percent=(-0.0625, 0.0625), rotate=(-45, 45), p=0.6),
        A.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15, hue=0.05, p=0.5),
        A.CoarseDropout(num_holes_range=(1, 8), hole_height_range=(8, 16), hole_width_range=(8, 16), fill=0, p=0.3)
    ])

    new_rows = []

    for index, row in tqdm(df.iterrows(), total=df.shape[0]):
        dx = row['dx']
        mult = multiplier_dict[dx]

        if mult == 0:
            continue
            
        img_id = row['image_id']
        img_path = os.path.join(train_dir, f"{img_id}.jpg")
        
        image = cv2.imread(img_path)
        if image is None:
            continue
            
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        for i in range(mult):
            augmented = transform(image=image)
            aug_img = augmented['image']
            
            new_img_id = f"{img_id}_aug_{i}"
            new_img_name = f"{new_img_id}.jpg"
            new_img_path = os.path.join(train_dir, new_img_name)
            
            cv2.imwrite(new_img_path, cv2.cvtColor(aug_img, cv2.COLOR_RGB2BGR))
            
            new_row = row.copy()
            new_row['image_id'] = new_img_id 
            new_rows.append(new_row)

    if new_rows:
        new_df = pd.DataFrame(new_rows)
        final_df = pd.concat([df, new_df], ignore_index=True)
        final_df = final_df.sample(frac=1, random_state=42).reset_index(drop=True)
        final_df.to_csv(output_csv_path, index=False)
    else:
        print("None")

if __name__ == "__main__":
    TRAIN_DIR = "HAM10000_Split_Dataset/images/train"
    CSV_INPUT = "HAM10000_Split_Dataset/train_meta.csv"
    CSV_OUTPUT = "HAM10000_Split_Dataset/train_meta_augmented.csv"
    
    capped_augmentation_ham10000(TRAIN_DIR, CSV_INPUT, CSV_OUTPUT)