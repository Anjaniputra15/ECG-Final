#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ECG Dataset Preparation Script
Creates metadata.csv and train/val/test splits for ECG PQRST detection
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path
import json
from sklearn.model_selection import train_test_split

def create_metadata_csv():
    """Create metadata.csv linking image-mask pairs"""
    
    # Define paths
    base_dir = Path(__file__).parent.parent
    images_dir = base_dir / "data" / "raw" / "scanned_ecgs"
    masks_dir = base_dir / "data" / "annotations" / "manual"
    
    # Get all image files
    image_files = sorted(list(images_dir.glob("*.png")))
    mask_files = sorted(list(masks_dir.glob("*.png")))
    
    print(f"Found {len(image_files)} images and {len(mask_files)} masks")
    
    # Create metadata list
    metadata = []
    
    for img_file in image_files:
        img_name = img_file.name
        mask_name = img_name  # Same naming pattern
        mask_file = masks_dir / mask_name
        
        if mask_file.exists():
            metadata.append({
                'image_id': img_name.replace('.png', ''),
                'image_path': f"data/raw/scanned_ecgs/{img_name}",
                'mask_path': f"data/annotations/manual/{mask_name}",
                'image_size': None,  # Will be filled during preprocessing
                'has_mask': True,
                'quality_score': None,  # Will be computed during preprocessing
                'leads_detected': None,  # Will be filled during lead extraction
                'processing_status': 'raw'
            })
        else:
            print(f"Warning: No mask found for {img_name}")
    
    # Create DataFrame
    df = pd.DataFrame(metadata)
    
    # Save metadata.csv
    metadata_path = base_dir / "data" / "raw" / "metadata.csv"
    df.to_csv(metadata_path, index=False)
    
    print(f"Created metadata.csv with {len(df)} entries")
    print(f"Saved to: {metadata_path}")
    
    return df

def create_train_val_test_splits(df, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    """Create train/validation/test splits"""
    
    base_dir = Path(__file__).parent.parent
    splits_dir = base_dir / "data" / "splits"
    
    # Verify ratios sum to 1
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1.0"
    
    # Get image IDs
    image_ids = df['image_id'].tolist()
    
    # First split: train vs (val + test)
    train_ids, temp_ids = train_test_split(
        image_ids, 
        test_size=(val_ratio + test_ratio),
        random_state=42,
        shuffle=True
    )
    
    # Second split: val vs test from temp
    val_ratio_adjusted = val_ratio / (val_ratio + test_ratio)
    val_ids, test_ids = train_test_split(
        temp_ids,
        test_size=(1 - val_ratio_adjusted),
        random_state=42,
        shuffle=True
    )
    
    # Create split dictionaries
    splits = {
        'train': train_ids,
        'val': val_ids,
        'test': test_ids
    }
    
    # Save splits to JSON files
    for split_name, ids in splits.items():
        split_data = {
            'split': split_name,
            'count': len(ids),
            'image_ids': ids
        }
        
        split_file = splits_dir / f"{split_name}.json"
        with open(split_file, 'w') as f:
            json.dump(split_data, f, indent=2)
        
        print(f"{split_name.capitalize()} split: {len(ids)} samples")
    
    return splits

def verify_data_integrity(df, splits):
    """Verify data integrity and correspondence"""
    
    base_dir = Path(__file__).parent.parent
    images_dir = base_dir / "data" / "raw" / "scanned_ecgs"
    masks_dir = base_dir / "data" / "annotations" / "manual"
    
    print("\n=== Data Integrity Check ===")
    
    # Check file existence
    missing_images = []
    missing_masks = []
    
    for _, row in df.iterrows():
        img_path = base_dir / row['image_path']
        mask_path = base_dir / row['mask_path']
        
        if not img_path.exists():
            missing_images.append(row['image_id'])
        
        if not mask_path.exists():
            missing_masks.append(row['image_id'])
    
    if missing_images:
        print(f"Missing images: {len(missing_images)}")
        print(f"   Examples: {missing_images[:5]}")
    else:
        print("All images exist - OK")
    
    if missing_masks:
        print(f"Missing masks: {len(missing_masks)}")
        print(f"   Examples: {missing_masks[:5]}")
    else:
        print("All masks exist - OK")
    
    # Check split integrity
    all_split_ids = []
    for split_name, ids in splits.items():
        all_split_ids.extend(ids)
    
    df_ids = set(df['image_id'])
    split_ids = set(all_split_ids)
    
    if df_ids == split_ids:
        print("All samples included in splits - OK")
    else:
        print(f"Split mismatch: {len(df_ids)} in metadata, {len(split_ids)} in splits")
    
    # Check for duplicates in splits
    if len(all_split_ids) == len(set(all_split_ids)):
        print("No duplicate samples across splits - OK")
    else:
        print("Duplicate samples found across splits")
    
    return len(missing_images) == 0 and len(missing_masks) == 0

def main():
    """Main dataset preparation pipeline"""
    
    print("Starting ECG Dataset Preparation")
    print("=" * 50)
    
    # Create metadata
    print("\nCreating metadata.csv...")
    df = create_metadata_csv()
    
    # Create splits
    print(f"\nCreating train/val/test splits...")
    splits = create_train_val_test_splits(df)
    
    # Verify integrity
    print(f"\nVerifying data integrity...")
    integrity_ok = verify_data_integrity(df, splits)
    
    # Summary
    print("\n" + "=" * 50)
    print("DATASET SUMMARY")
    print("=" * 50)
    print(f"Total samples: {len(df)}")
    print(f"Train samples: {len(splits['train'])} ({len(splits['train'])/len(df)*100:.1f}%)")
    print(f"Val samples: {len(splits['val'])} ({len(splits['val'])/len(df)*100:.1f}%)")
    print(f"Test samples: {len(splits['test'])} ({len(splits['test'])/len(df)*100:.1f}%)")
    print(f"Data integrity: {'PASSED' if integrity_ok else 'FAILED'}")
    
    if integrity_ok:
        print("\nDataset preparation completed successfully!")
        print("Ready for Phase 2: Preprocessing")
    else:
        print("\nPlease fix data integrity issues before proceeding")
    
    return integrity_ok

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)