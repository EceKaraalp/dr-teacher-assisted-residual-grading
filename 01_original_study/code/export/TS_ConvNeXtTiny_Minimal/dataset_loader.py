"""
Dataset Loader for APTOS2019 Blindness Detection Dataset (PyTorch)
Handles loading images with retry logic for robustness
"""

import os
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, List
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import time


class APTOS2019Dataset(Dataset):
    """PyTorch Dataset for APTOS2019."""
    
    def __init__(self, images: np.ndarray, labels: np.ndarray, num_classes: int = 5):
        """
        Args:
            images: (N, 224, 224, 3) uint8 numpy array
            labels: (N,) int32 numpy array
            num_classes: Number of classes for one-hot encoding
        """
        self.images = images
        self.labels = labels
        self.num_classes = num_classes
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        # Image: (224, 224, 3) uint8 -> (3, 224, 224) float32 tensor [0-255 uint8 range kept for model normalization]
        image = self.images[idx]  # (224, 224, 3) uint8
        image = torch.from_numpy(image).permute(2, 0, 1).float()  # (3, 224, 224)
        
        # Label: one-hot encoding
        label = np.zeros(self.num_classes, dtype=np.float32)
        label[self.labels[idx]] = 1.0
        label = torch.from_numpy(label)
        
        return image, label


class APTOS2019DatasetLoader:
    """
    Loads APTOS2019 dataset with automatic retry logic.
    No preprocessing is applied - only resizing for tensor compatibility.
    """
    
    def __init__(self, dataset_path: str = "APTOS2019", image_size: int = 224, max_retries: int = 3):
        self.dataset_path = Path(dataset_path)
        self.image_size = image_size
        self.max_retries = max_retries
        self.train_images_path = self.dataset_path / "train_images"
        self.test_images_path = self.dataset_path / "test_images"
        
        if not self.dataset_path.exists():
            raise FileNotFoundError(f"Dataset path not found: {self.dataset_path}")
    
    def _load_image_with_retry(self, image_path: str) -> np.ndarray:
        """Load image with automatic retry on failure."""
        for attempt in range(self.max_retries):
            try:
                image = Image.open(image_path).convert('RGB')
                image = image.resize((self.image_size, self.image_size), Image.BILINEAR)
                image = np.array(image, dtype=np.uint8)
                return image
                
            except Exception as e:
                if attempt < self.max_retries - 1:
                    wait_time = 2 ** attempt
                    print(f"Attempt {attempt + 1} failed: {str(e)}. Retrying in {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    print(f"Failed to load image after {self.max_retries} attempts: {image_path}")
                    raise
    
    def load_train_validation_test_split(self, 
                                        train_ratio: float = 0.8,
                                        val_ratio: float = 0.1,
                                        test_ratio: float = 0.1) -> Tuple:
        """Load APTOS2019 training data and split into train/validation/test."""
        train_csv_path = self.dataset_path / "train.csv"
        if not train_csv_path.exists():
            raise FileNotFoundError(f"Training CSV not found: {train_csv_path}")
        
        df_train = pd.read_csv(train_csv_path)
        
        print(f"Total training samples: {len(df_train)}")
        print(f"Class distribution:\n{df_train['diagnosis'].value_counts().sort_index()}")
        
        # Kullanıcının resmi yüklemeden önce kaça kaç bölündüğünü görmesi için bilgi ekranı
        total_samples = len(df_train)
        print("\n--- Öngörülen Veri Dağılımı ---")
        print(f"Eğitim (Train) : ~{int(total_samples * train_ratio)} örnek (%{train_ratio * 100:.1f})")
        print(f"Doğrulama (Val): ~{int(total_samples * val_ratio)} örnek (%{val_ratio * 100:.1f})")
        print(f"Test (Test)    : ~{int(total_samples * test_ratio)} örnek (%{test_ratio * 100:.1f})")
        print("-------------------------------\nResimler diske yüklenmeye başlıyor, lütfen bekleyin...\n")
        
        images = []
        labels = []
        
        for idx, row in df_train.iterrows():
            image_id = row['id_code']
            label = row['diagnosis']
            
            image_files = list(self.train_images_path.glob(f"{image_id}*"))
            
            if not image_files:
                print(f"Warning: Image not found for {image_id}")
                continue
            
            image_path = str(image_files[0])
            
            try:
                image = self._load_image_with_retry(image_path)
                images.append(image)
                labels.append(label)
            except Exception as e:
                print(f"Skipping image {image_id}: {str(e)}")
                continue
            
            if (idx + 1) % 100 == 0:
                print(f"Loaded {idx + 1}/{len(df_train)} images")
        
        images = np.array(images, dtype=np.uint8)
        labels = np.array(labels, dtype=np.int32)
        
        print(f"Successfully loaded {len(images)} images")
        
        # Split into train, validation, test
        # test_ratio oranında veriyi direkt test seti olarak ayırıyoruz
        X_temp, X_test, y_temp, y_test = train_test_split(
            images, labels, 
            test_size=test_ratio,
            random_state=42, 
            stratify=labels
        )
        
        # Kalan veriyi (X_temp), istenen validation oranına ulaşacak şekilde bölüyoruz.
        # Temp verisinin ne kadarının val_ratio'ya denk geldiğini bulmalıyız: val_ratio / (train_ratio + val_ratio)
        val_split_from_temp = val_ratio / (train_ratio + val_ratio)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp,
            test_size=val_split_from_temp,
            random_state=42,
            stratify=y_temp
        )
        
        print(f"Train set: {len(X_train)} samples")
        print(f"Validation set: {len(X_val)} samples")
        print(f"Test set: {len(X_test)} samples")
        
        return X_train, y_train, X_val, y_val, X_test, y_test
    
    def create_dataloader(self, 
                         images: np.ndarray, 
                         labels: np.ndarray,
                         batch_size: int = 16,
                         shuffle: bool = True,
                         num_workers: int = 0) -> DataLoader:
        """Create PyTorch DataLoader from images and labels."""
        dataset = APTOS2019Dataset(images, labels, num_classes=5)
        dataloader = DataLoader(
            dataset, batch_size=batch_size, shuffle=shuffle,
            num_workers=num_workers, pin_memory=True, drop_last=False
        )
        return dataloader


def get_data_loaders(dataset_path: str = "APTOS2019", 
                    batch_size: int = 16) -> Tuple:
    """
    Convenience function to get all data loaders.
    
    Returns:
        Tuple: (train_loader, val_loader, test_loader, class_weights,
                (X_train, y_train, X_val, y_val, X_test, y_test))
    """
    loader = APTOS2019DatasetLoader(dataset_path)
    X_train, y_train, X_val, y_val, X_test, y_test = loader.load_train_validation_test_split()
    
    train_loader = loader.create_dataloader(X_train, y_train, batch_size, shuffle=True)
    val_loader = loader.create_dataloader(X_val, y_val, batch_size, shuffle=False)
    test_loader = loader.create_dataloader(X_test, y_test, batch_size, shuffle=False)
    
    # Calculate class weights
    unique, counts = np.unique(y_train, return_counts=True)
    class_weights = dict(zip(unique.tolist(), (len(y_train) / (len(unique) * counts)).astype(float).tolist()))
    
    print(f"Class weights: {class_weights}")
    
    return train_loader, val_loader, test_loader, class_weights, (X_train, y_train, X_val, y_val, X_test, y_test)
