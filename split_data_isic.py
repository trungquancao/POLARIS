import os
import shutil
import random
from tqdm import tqdm

def create_yolo_structure(base_dir):
    dirs = [
        os.path.join(base_dir, 'images', 'train'),
        os.path.join(base_dir, 'images', 'val'),
        os.path.join(base_dir, 'images', 'test'),
        os.path.join(base_dir, 'masks', 'train'),
        os.path.join(base_dir, 'masks', 'val'),
        os.path.join(base_dir, 'masks', 'test')
    ]
    for d in dirs:
        os.makedirs(d, exist_ok=True)
    return dirs

def split_and_copy_dataset(img_dir, mask_dir, output_dir, split_ratio=(0.8, 0.1, 0.1), seed=42):
    random.seed(seed)
    
    create_yolo_structure(output_dir)
    
    valid_images = [f for f in os.listdir(img_dir) if f.endswith('.jpg') or f.endswith('.png')]
    image_ids = [f.split('.')[0] for f in valid_images]
    
    random.shuffle(image_ids)

    total_imgs = len(image_ids)
    train_count = int(total_imgs * split_ratio[0])
    val_count = int(total_imgs * split_ratio[1])

    train_ids = image_ids[:train_count]
    val_ids = image_ids[train_count : train_count + val_count]
    test_ids = image_ids[train_count + val_count:] # Phần còn lại
    
    def copy_files(ids_list, subset_name):
        for img_id in tqdm(ids_list, desc=f"Copying {subset_name.upper()}"):
            img_filename = f"{img_id}.jpg" 
            mask_filename = f"{img_id}_segmentation.png"
            
            src_img = os.path.join(img_dir, img_filename)
            src_mask = os.path.join(mask_dir, mask_filename)

            dst_img = os.path.join(output_dir, 'images', subset_name, img_filename)
            dst_mask = os.path.join(output_dir, 'masks', subset_name, mask_filename)
            
            if os.path.exists(src_img) and os.path.exists(src_mask):
                shutil.copy2(src_img, dst_img)
                shutil.copy2(src_mask, dst_mask)
            else:
                print(f"Cant not find Mask for ID {img_id}")

    copy_files(train_ids, 'train')
    copy_files(val_ids, 'val')
    copy_files(test_ids, 'test')


if __name__ == "__main__":
    IMG_INPUT = "ISIC2018_Task1-2_Training_Input_Color_Processed" 
    MASK_INPUT = "ISIC2018_Task1_Training_GroundTruth" 
    OUTPUT_SPLIT = "ISIC_Split_Dataset"

    split_and_copy_dataset(IMG_INPUT, MASK_INPUT, OUTPUT_SPLIT, split_ratio=(0.8, 0.1, 0.1), seed=42)
