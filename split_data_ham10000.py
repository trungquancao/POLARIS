import os
import shutil
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm

def split_ham10000_by_lesion(img_dir, csv_path, output_dir, seed=42):
    df = pd.read_csv(csv_path)

    unique_lesions = df['lesion_id'].unique()
    train_lesions, temp_lesions = train_test_split(unique_lesions, test_size=0.2, random_state=seed)
    val_lesions, test_lesions = train_test_split(temp_lesions, test_size=0.5, random_state=seed)
    
    train_df = df[df['lesion_id'].isin(train_lesions)].copy()
    val_df = df[df['lesion_id'].isin(val_lesions)].copy()
    test_df = df[df['lesion_id'].isin(test_lesions)].copy()

    for subset in ['train', 'val', 'test']:
        os.makedirs(os.path.join(output_dir, 'images', subset), exist_ok=True)
        
    train_df.to_csv(os.path.join(output_dir, 'train_meta.csv'), index=False)
    val_df.to_csv(os.path.join(output_dir, 'val_meta.csv'), index=False)
    test_df.to_csv(os.path.join(output_dir, 'test_meta.csv'), index=False)

    def copy_images(subset_df, subset_name):
        dest_dir = os.path.join(output_dir, 'images', subset_name)
        for img_id in tqdm(subset_df['image_id'], desc=f"Copying {subset_name.upper()}"):
            img_filename = f"{img_id}.jpg"
            src_path = os.path.join(img_dir, img_filename)
            dst_path = os.path.join(dest_dir, img_filename)
            
            if os.path.exists(src_path):
                shutil.copy2(src_path, dst_path)
            else:
                print(f"Cant not find img {img_filename}")

    copy_images(train_df, 'train')
    copy_images(val_df, 'val')
    copy_images(test_df, 'test')
    

if __name__ == "__main__":
    IMG_INPUT = "HAM10000" 
    CSV_INPUT = "HAM10000_metadata.csv" 
    OUTPUT_SPLIT = "HAM10000_Split_Dataset"
    
    split_ham10000_by_lesion(IMG_INPUT, CSV_INPUT, OUTPUT_SPLIT, seed=42)