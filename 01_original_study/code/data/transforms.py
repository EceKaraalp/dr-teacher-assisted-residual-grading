"""
Data Transforms with CLAHE and Augmentations
"""
import cv2
import numpy as np
from torchvision import transforms

class CLAHETransform:
    def __init__(self, clip_limit=2.0, tile_grid_size=(8, 8)):
        self.clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)

    def __call__(self, img):
        # img is expected to be a numpy array (cv2 format)
        if isinstance(img, np.ndarray):
            lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
            lab[..., 0] = self.clahe.apply(lab[..., 0])
            img_clahe = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
            return img_clahe
        return img  # Return original if not numpy

def get_transforms(phase='train', img_size=224):
    if phase == 'train':
        return transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(20),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])