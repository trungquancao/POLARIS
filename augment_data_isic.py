import os
import cv2
import albumentations as A
from tqdm import tqdm

def augment_isic_segmentation(train_img_dir, train_mask_dir, aug_multiplier=3):
    transform = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.Affine(scale=(0.85, 1.15), translate_percent=(-0.05, 0.05), rotate=(-45, 45), p=0.7),
        A.ElasticTransform(alpha=1, sigma=50, alpha_affine=50, p=0.3),
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05, p=0.5),
    ])

    image_files = [f for f in os.listdir(train_img_dir) if f.endswith('.jpg') and '_aug_' not in f]

    for img_name in tqdm(image_files, desc="Augmenting ISIC Train Set"):
        img_path = os.path.join(train_img_dir, img_name)
        
        base_name = img_name.split('.')[0]
        mask_name = f"{base_name}_segmentation.png"
        mask_path = os.path.join(train_mask_dir, mask_name)

        if not os.path.exists(mask_path):
            continue

        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        for i in range(aug_multiplier):
            augmented = transform(image=image, mask=mask)
            aug_img = augmented['image']
            aug_mask = augmented['mask']

            new_img_name = f"{base_name}_aug_{i}.jpg"
            new_mask_name = f"{base_name}_segmentation_aug_{i}.png"

            cv2.imwrite(os.path.join(train_img_dir, new_img_name), cv2.cvtColor(aug_img, cv2.COLOR_RGB2BGR))
            cv2.imwrite(os.path.join(train_mask_dir, new_mask_name), aug_mask)

    final_img_count = len(os.listdir(train_img_dir))
    final_mask_count = len(os.listdir(train_mask_dir))
    assert final_img_count == final_mask_count, "Error"

if __name__ == "__main__":
    TRAIN_IMG_DIR = "ISIC_Split_Dataset/images/train"
    TRAIN_MASK_DIR = "ISIC_Split_Dataset/masks/train"
    
    augment_isic_segmentation(TRAIN_IMG_DIR, TRAIN_MASK_DIR, aug_multiplier=3)