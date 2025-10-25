"""
Data download and preprocessing module for waste classification.
Handles dataset acquisition, image preprocessing, and train/val/test splitting.
"""

import os
import shutil
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
import cv2
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import json


class WasteDataLoader:
    """Handles data acquisition, preprocessing, and organization."""
    
    def __init__(self, raw_dir='data/raw', processed_dir='data/processed'):
        self.raw_dir = Path(raw_dir)
        self.processed_dir = Path(processed_dir)
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        
        # Image preprocessing parameters
        self.img_size = (224, 224)  # MobileNetV2 input size
        self.channels = 3
        
    def scan_directory_structure(self, root_path):
        """
        Scan directory structure and create labels DataFrame.
        Assumes structure: root/class_name/image.jpg
        """
        data_records = []
        root = Path(root_path)
        
        for class_dir in sorted(root.iterdir()):
            if not class_dir.is_dir():
                continue
                
            class_name = class_dir.name
            for img_path in class_dir.glob('*.*'):
                if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
                    data_records.append({
                        'path': str(img_path),
                        'filename': img_path.name,
                        'class': class_name
                    })
        
        df = pd.DataFrame(data_records)
        print(f"\nFound {len(df)} images across {df['class'].nunique()} classes")
        print(df['class'].value_counts())
        
        return df
    
    def create_class_mapping(self, df, merge_classes=None):
        """
        Create numerical class mapping and optionally merge classes.
        
        Args:
            df: DataFrame with 'class' column
            merge_classes: Dict mapping original classes to target classes
                          e.g., {'cardboard': 'recyclable', 'paper': 'recyclable'}
        """
        if merge_classes:
            df['class'] = df['class'].map(lambda x: merge_classes.get(x, x))
        
        unique_classes = sorted(df['class'].unique())
        class_to_idx = {cls: idx for idx, cls in enumerate(unique_classes)}
        idx_to_class = {idx: cls for cls, idx in class_to_idx.items()}
        
        df['label'] = df['class'].map(class_to_idx)
        
        # Save mapping
        mapping_path = self.processed_dir / 'class_mapping.json'
        with open(mapping_path, 'w') as f:
            json.dump({
                'class_to_idx': class_to_idx,
                'idx_to_class': idx_to_class
            }, f, indent=2)
        
        print(f"\nClass mapping saved to {mapping_path}")
        print(f"Classes: {class_to_idx}")
        
        return df, class_to_idx, idx_to_class
    
    def preprocess_image(self, img_path, augment=False):
        """
        Load and preprocess single image.
        
        Args:
            img_path: Path to image file
            augment: Whether to apply augmentation
        """
        try:
            # Load image
            img = cv2.imread(str(img_path))
            if img is None:
                return None
            
            # Convert BGR to RGB
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Resize
            img = cv2.resize(img, self.img_size)
            
            # Augmentation (basic)
            if augment:
                # Random horizontal flip
                if np.random.rand() > 0.5:
                    img = cv2.flip(img, 1)
                
                # Random brightness adjustment
                if np.random.rand() > 0.5:
                    factor = np.random.uniform(0.8, 1.2)
                    img = np.clip(img * factor, 0, 255).astype(np.uint8)
                
                # Random rotation (small angle)
                if np.random.rand() > 0.5:
                    angle = np.random.uniform(-15, 15)
                    h, w = img.shape[:2]
                    M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1.0)
                    img = cv2.warpAffine(img, M, (w, h))
            
            # Normalize to [0, 1]
            img = img.astype(np.float32) / 255.0
            
            return img
            
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
            return None
    
    def prepare_splits(self, df, test_size=0.15, val_size=0.15, random_state=42):
        """
        Split data into train/val/test sets with stratification.
        """
        # First split: separate test set
        train_val_df, test_df = train_test_split(
            df, 
            test_size=test_size, 
            stratify=df['label'],
            random_state=random_state
        )
        
        # Second split: separate validation from training
        val_size_adjusted = val_size / (1 - test_size)
        train_df, val_df = train_test_split(
            train_val_df,
            test_size=val_size_adjusted,
            stratify=train_val_df['label'],
            random_state=random_state
        )
        
        print(f"\nDataset splits:")
        print(f"Train: {len(train_df)} ({len(train_df)/len(df)*100:.1f}%)")
        print(f"Val:   {len(val_df)} ({len(val_df)/len(df)*100:.1f}%)")
        print(f"Test:  {len(test_df)} ({len(test_df)/len(df)*100:.1f}%)")
        
        # Save split information
        train_df.to_csv(self.processed_dir / 'train.csv', index=False)
        val_df.to_csv(self.processed_dir / 'val.csv', index=False)
        test_df.to_csv(self.processed_dir / 'test.csv', index=False)
        
        return train_df, val_df, test_df
    
    def create_preprocessed_dataset(self, df, split_name, augment=False):
        """
        Preprocess all images and save to disk.
        """
        output_dir = self.processed_dir / split_name
        output_dir.mkdir(exist_ok=True)
        
        successful = 0
        failed = 0
        
        print(f"\nProcessing {split_name} set ({len(df)} images)...")
        
        for idx, row in tqdm(df.iterrows(), total=len(df)):
            img = self.preprocess_image(row['path'], augment=augment)
            
            if img is not None:
                # Save as NPY for faster loading
                save_path = output_dir / f"{row['filename']}.npy"
                np.save(save_path, img)
                successful += 1
            else:
                failed += 1
        
        print(f"Processed: {successful} successful, {failed} failed")
        
        return successful, failed


def main():
    """Main execution function."""
    loader = WasteDataLoader()
    
    # Scan TRAIN directory
    print("Scanning TRAIN dataset directory...")
    train_path = 'data/raw/TRAIN'
    
    if not os.path.exists(train_path):
        print(f"Error: {train_path} not found!")
        return
    
    train_df = loader.scan_directory_structure(train_path)
    
    # Scan TEST directory
    print("\nScanning TEST dataset directory...")
    test_path = 'data/raw/TEST'
    
    if not os.path.exists(test_path):
        print(f"Error: {test_path} not found!")
        return
    
    test_df = loader.scan_directory_structure(test_path)
    
    # Create class mapping (O=Organic, R=Recyclable)
    # Combine both dataframes to ensure consistent mapping
    combined_df = pd.concat([train_df, test_df], ignore_index=True)
    combined_df, class_to_idx, idx_to_class = loader.create_class_mapping(
        combined_df, 
        merge_classes=None  # Keep original O/R classes
    )
    
    # Split combined data back
    train_df = combined_df[combined_df['path'].str.contains('/TRAIN/')].reset_index(drop=True)
    test_df = combined_df[combined_df['path'].str.contains('/TEST/')].reset_index(drop=True)
    
    # Split TRAIN into train/val (85/15 split since we already have TEST)
    from sklearn.model_selection import train_test_split
    train_df, val_df = train_test_split(
        train_df,
        test_size=0.15,
        stratify=train_df['label'],
        random_state=42
    )
    
    print(f"\n Final splits:")
    print(f"Train: {len(train_df)} ({len(train_df)/len(combined_df)*100:.1f}%)")
    print(f"Val:   {len(val_df)} ({len(val_df)/len(combined_df)*100:.1f}%)")
    print(f"Test:  {len(test_df)} ({len(test_df)/len(combined_df)*100:.1f}%)")
    
    # Save split CSVs
    train_df.to_csv(loader.processed_dir / 'train.csv', index=False)
    val_df.to_csv(loader.processed_dir / 'val.csv', index=False)
    test_df.to_csv(loader.processed_dir / 'test.csv', index=False)
    
    print(f"\n Data preparation complete!")
    print(f"Metadata saved in {loader.processed_dir}")
    print(f"\nClass mapping: {class_to_idx}")


if __name__ == '__main__':
    main()