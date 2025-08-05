#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simple ECG Dataset Preparation Script (No pandas/sklearn dependencies)
Creates metadata.csv and train/val/test splits for ECG PQRST detection
"""

import os
import json
import random
from pathlib import Path

def create_metadata_csv():
    """Create metadata.csv linking image-mask pairs"""
    
    # Define paths
    base_dir = Path(__file__).parent.parent
    images_dir = base_dir / "data" / "raw" / "scanned_ecgs"
    masks_dir = base_dir / "data" / "annotations" / "manual"
    
    # Get all image files
    image_files = sorted([f for f in images_dir.glob("*.png")])
    mask_files = sorted([f for f in masks_dir.glob("*.png")])
    
    print(f"Found {len(image_files)} images and {len(mask_files)} masks")
    
    # Create metadata list
    metadata_lines = ["image_id,image_path,mask_path,has_mask,processing_status"]
    
    for img_file in image_files:
        img_name = img_file.name
        mask_name = img_name  # Same naming pattern
        mask_file = masks_dir / mask_name
        
        if mask_file.exists():
            image_id = img_name.replace('.png', '')
            metadata_lines.append(
                f"{image_id},data/raw/scanned_ecgs/{img_name},data/annotations/manual/{mask_name},True,raw"
            )
        else:
            print(f"Warning: No mask found for {img_name}")
    
    # Save metadata.csv
    metadata_path = base_dir / "data" / "raw" / "metadata.csv"
    with open(metadata_path, 'w') as f:
        f.write('\n'.join(metadata_lines))
    
    print(f"Created metadata.csv with {len(metadata_lines)-1} entries")
    print(f"Saved to: {metadata_path}")
    
    return [line.split(',')[0] for line in metadata_lines[1:]]  # Return image IDs

def create_train_val_test_splits(image_ids, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    """Create train/validation/test splits"""
    
    base_dir = Path(__file__).parent.parent
    splits_dir = base_dir / "data" / "splits"
    
    # Shuffle the image IDs
    random.seed(42)
    shuffled_ids = image_ids.copy()
    random.shuffle(shuffled_ids)
    
    # Calculate split sizes
    total = len(shuffled_ids)
    train_size = int(total * train_ratio)
    val_size = int(total * val_ratio)
    
    # Create splits
    train_ids = shuffled_ids[:train_size]
    val_ids = shuffled_ids[train_size:train_size + val_size]
    test_ids = shuffled_ids[train_size + val_size:]
    
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

def verify_data_integrity(image_ids, splits):
    """Verify data integrity and correspondence"""
    
    base_dir = Path(__file__).parent.parent
    images_dir = base_dir / "data" / "raw" / "scanned_ecgs"
    masks_dir = base_dir / "data" / "annotations" / "manual"
    
    print("\n=== Data Integrity Check ===")
    
    # Check file existence
    missing_images = []
    missing_masks = []
    
    for image_id in image_ids:
        img_path = images_dir / f"{image_id}.png"
        mask_path = masks_dir / f"{image_id}.png"
        
        if not img_path.exists():
            missing_images.append(image_id)
        
        if not mask_path.exists():
            missing_masks.append(image_id)
    
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
    
    if set(image_ids) == set(all_split_ids):
        print("All samples included in splits - OK")
    else:
        print(f"Split mismatch: {len(image_ids)} in metadata, {len(all_split_ids)} in splits")
    
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
    image_ids = create_metadata_csv()
    
    # Create splits
    print(f"\nCreating train/val/test splits...")
    splits = create_train_val_test_splits(image_ids)
    
    # Verify integrity
    print(f"\nVerifying data integrity...")
    integrity_ok = verify_data_integrity(image_ids, splits)
    
    # Summary
    print("\n" + "=" * 50)
    print("DATASET SUMMARY")
    print("=" * 50)
    print(f"Total samples: {len(image_ids)}")
    print(f"Train samples: {len(splits['train'])} ({len(splits['train'])/len(image_ids)*100:.1f}%)")
    print(f"Val samples: {len(splits['val'])} ({len(splits['val'])/len(image_ids)*100:.1f}%)")
    print(f"Test samples: {len(splits['test'])} ({len(splits['test'])/len(image_ids)*100:.1f}%)")
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