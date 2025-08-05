#!/usr/bin/env python3
"""
Quick setup script to create training data structure
"""

import os
from pathlib import Path
import pandas as pd
import numpy as np

print("🔧 Setting up training data structure...")
print("="*50)

# Create data directory structure
base_dir = Path("data")
base_dir.mkdir(exist_ok=True)

# Create subdirectories
subdirs = [
    "raw/scanned_ecgs",
    "annotations/manual", 
    "processed/train",
    "processed/val",
    "processed/test"
]

for subdir in subdirs:
    (base_dir / subdir).mkdir(parents=True, exist_ok=True)
    print(f"✅ Created directory: data/{subdir}")

# Create metadata file
print("\n📄 Creating metadata.csv...")

# Check if original data exists
original_imgs = Path("imgs")
original_masks = Path("masks")

samples = []

if original_imgs.exists() and original_masks.exists():
    print(f"   Found original data: {len(list(original_imgs.glob('*.png')))} images")
    
    # Use actual files
    img_files = sorted(list(original_imgs.glob("*.png")))
    mask_files = sorted(list(original_masks.glob("*.png")))
    
    print(f"   Images: {len(img_files)}, Masks: {len(mask_files)}")
    
    # Create balanced splits
    total_samples = min(len(img_files), len(mask_files))
    
    # Calculate splits (70/15/15)
    train_size = int(total_samples * 0.7)
    val_size = int(total_samples * 0.15)
    test_size = total_samples - train_size - val_size
    
    print(f"   Split: Train={train_size}, Val={val_size}, Test={test_size}")
    
    # Assign splits
    splits = ['train'] * train_size + ['val'] * val_size + ['test'] * test_size
    
    for i in range(total_samples):
        if i < len(img_files) and i < len(mask_files):
            samples.append({
                'image_path': f"raw/scanned_ecgs/{img_files[i].name}",
                'mask_path': f"annotations/manual/{mask_files[i].name}",
                'split': splits[i]
            })
    
    # Copy files to proper structure
    print("\n📁 Copying files to proper structure...")
    
    import shutil
    
    for sample in samples:
        src_img = original_imgs / Path(sample['image_path']).name
        dst_img = base_dir / sample['image_path']
        
        src_mask = original_masks / Path(sample['mask_path']).name  
        dst_mask = base_dir / sample['mask_path']
        
        if src_img.exists() and not dst_img.exists():
            shutil.copy2(src_img, dst_img)
        
        if src_mask.exists() and not dst_mask.exists():
            shutil.copy2(src_mask, dst_mask)
    
    print(f"   ✅ Copied {len(samples)} image-mask pairs")

else:
    print("   No original data found, creating dummy metadata...")
    
    # Create dummy metadata for testing
    for i in range(200):
        filename = f"ecg_{i:04d}.png"
        
        # Assign to splits (140 train, 30 val, 30 test)
        if i < 140:
            split = "train"
        elif i < 170:
            split = "val"
        else:
            split = "test"
        
        samples.append({
            'image_path': f"raw/scanned_ecgs/{filename}",
            'mask_path': f"annotations/manual/{filename}",
            'split': split
        })

# Create metadata CSV
metadata_df = pd.DataFrame(samples)
metadata_path = base_dir / "metadata.csv"
metadata_df.to_csv(metadata_path, index=False)

print(f"\n✅ Created metadata.csv with {len(samples)} samples")
print(f"   Location: {metadata_path}")

# Print summary
if len(samples) > 0:
    split_counts = metadata_df['split'].value_counts()
    print(f"\n📊 Data splits:")
    for split, count in split_counts.items():
        print(f"   {split}: {count} samples ({count/len(samples)*100:.1f}%)")

# Create dummy images if needed
print("\n🖼️  Creating dummy ECG images for testing...")

try:
    import cv2
    import numpy as np
    
    # Create dummy ECG images
    for sample in samples[:10]:  # Just create first 10 for testing
        img_path = base_dir / sample['image_path']
        mask_path = base_dir / sample['mask_path']
        
        if not img_path.exists():
            # Create realistic ECG-like image
            img = np.ones((512, 512, 3), dtype=np.uint8) * 255  # White background
            
            # Add grid lines
            for i in range(0, 512, 25):
                cv2.line(img, (i, 0), (i, 512), (200, 200, 200), 1)
                cv2.line(img, (0, i), (512, i), (200, 200, 200), 1)
            
            # Add ECG trace
            y_center = 256
            for x in range(512):
                # Simple ECG-like wave
                y = y_center + int(30 * np.sin(x * 0.02) + 10 * np.sin(x * 0.1))
                if 0 <= y < 512:
                    cv2.circle(img, (x, y), 2, (0, 0, 0), -1)
            
            img_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(img_path), img)
        
        if not mask_path.exists():
            # Create mask highlighting ECG trace
            mask = np.zeros((512, 512), dtype=np.uint8)
            
            # Mark ECG trace area
            y_center = 256
            for x in range(512):
                y = y_center + int(30 * np.sin(x * 0.02) + 10 * np.sin(x * 0.1))
                if 0 <= y < 512:
                    cv2.circle(mask, (x, y), 5, 255, -1)
            
            mask_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(mask_path), mask)
    
    print(f"   ✅ Created {min(10, len(samples))} dummy ECG images")

except ImportError:
    print("   ⚠️  OpenCV not available, skipping dummy image creation")
    print("   Install with: pip install opencv-python")

print("\n" + "="*50)
print("🎉 Training data setup complete!")
print("\nNext steps:")
print("1. Run training: python training/advanced_trainer.py")
print("2. Or test with: python test_training.py")
print("\nFile structure created:")
print("data/")
print("├── metadata.csv")
print("├── raw/scanned_ecgs/")
print("├── annotations/manual/")
print("└── processed/train|val|test/")
print("\n" + "="*50)