#!/usr/bin/env python3
"""
Mask R-CNN ECG Trainer
Training pipeline for ECG wave detection and segmentation.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision.transforms import functional as F
import numpy as np
import cv2
import json
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Import our models and annotation system
import sys
sys.path.append(str(Path(__file__).parent.parent))
from models.maskrcnn_ecg import ECGMaskRCNN, create_ecg_maskrcnn
from annotation.ecg_wave_annotator import ECGWaveAnnotator, create_ecg_annotations


class ECGWaveDataset(Dataset):
    """
    Dataset for ECG wave detection and segmentation training.
    Loads ECG images with P, QRS, T wave annotations.
    """
    
    def __init__(self, data_path: str, split: str = 'train', transform=None):
        self.data_path = Path(data_path)
        self.transform = transform
        self.split = split
        
        # Load image paths
        images_dir = self.data_path / 'processed/ptbxl/images'
        self.image_paths = list(images_dir.glob('*.png'))
        
        # Create or load annotations
        self.annotations = self._load_or_create_annotations()
        
        print(f"Loaded {len(self.image_paths)} ECG images for {split}")
        
    def _load_or_create_annotations(self) -> dict:
        """Load existing annotations or create new ones."""
        annotations_path = self.data_path / 'processed/ptbxl/wave_annotations.json'
        
        if annotations_path.exists():
            print("Loading existing wave annotations...")
            with open(annotations_path, 'r') as f:
                return json.load(f)
        else:
            print("Creating new wave annotations...")
            return self._create_wave_annotations(annotations_path)
    
    def _create_wave_annotations(self, save_path: Path) -> dict:
        """Create wave annotations for all ECG images."""
        annotator = ECGWaveAnnotator()
        all_annotations = {'images': [], 'annotations': [], 'categories': []}
        
        # Categories
        categories = [
            {'id': 1, 'name': 'P_wave', 'supercategory': 'wave'},
            {'id': 2, 'name': 'QRS_complex', 'supercategory': 'wave'},
            {'id': 3, 'name': 'T_wave', 'supercategory': 'wave'}
        ]
        all_annotations['categories'] = categories
        
        annotation_id = 1
        
        # Process each image
        for image_id, image_path in enumerate(tqdm(self.image_paths, desc="Creating annotations")):
            try:
                # Create annotation for this image
                image_ann = create_ecg_annotations(str(image_path))
                
                # Add image info
                image_info = image_ann['images'][0]
                image_info['id'] = image_id + 1
                all_annotations['images'].append(image_info)
                
                # Add annotations with updated IDs
                for ann in image_ann['annotations']:
                    ann['id'] = annotation_id
                    ann['image_id'] = image_id + 1
                    all_annotations['annotations'].append(ann)
                    annotation_id += 1
                    
            except Exception as e:
                print(f"Error processing {image_path}: {e}")
                continue
        
        # Save annotations
        with open(save_path, 'w') as f:
            json.dump(all_annotations, f, indent=2)
        
        print(f"Created {len(all_annotations['annotations'])} wave annotations")
        return all_annotations
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        # Load image
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert('RGB')
        
        # Find annotations for this image
        image_name = image_path.name
        image_id = None
        
        # Find image ID
        for img_info in self.annotations['images']:
            if img_info['file_name'] == image_name:
                image_id = img_info['id']
                break
        
        if image_id is None:
            # Return empty targets if no annotations
            targets = {
                'boxes': torch.zeros((0, 4), dtype=torch.float32),
                'labels': torch.zeros((0,), dtype=torch.int64),
                'masks': torch.zeros((0, image.height, image.width), dtype=torch.uint8),
                'image_id': torch.tensor([idx]),
                'area': torch.zeros((0,), dtype=torch.float32),
                'iscrowd': torch.zeros((0,), dtype=torch.int64)
            }
        else:
            # Get annotations for this image
            image_annotations = [
                ann for ann in self.annotations['annotations'] 
                if ann['image_id'] == image_id
            ]
            
            targets = self._process_annotations(image_annotations, image.width, image.height)
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        return image, targets
    
    def _process_annotations(self, annotations, img_width, img_height):
        """Convert annotations to PyTorch tensors."""
        boxes = []
        labels = []
        masks = []
        areas = []
        iscrowds = []
        
        for ann in annotations:
            # Bounding box [x, y, width, height] -> [x1, y1, x2, y2]
            x, y, w, h = ann['bbox']
            boxes.append([x, y, x + w, y + h])
            
            # Label
            labels.append(ann['category_id'])
            
            # Area
            areas.append(ann['area'])
            
            # Crowd
            iscrowds.append(ann['iscrowd'])
            
            # Mask
            if 'segmentation' in ann and ann['segmentation']:
                mask = self._create_mask_from_segmentation(
                    ann['segmentation'], img_width, img_height
                )
            else:
                # Create mask from bounding box if segmentation not available
                mask = np.zeros((img_height, img_width), dtype=np.uint8)
                mask[y:y+h, x:x+w] = 1
            
            masks.append(mask)
        
        # Convert to tensors
        if len(boxes) == 0:
            # Empty annotations
            targets = {
                'boxes': torch.zeros((0, 4), dtype=torch.float32),
                'labels': torch.zeros((0,), dtype=torch.int64),
                'masks': torch.zeros((0, img_height, img_width), dtype=torch.uint8),
                'image_id': torch.tensor([0]),
                'area': torch.zeros((0,), dtype=torch.float32),
                'iscrowd': torch.zeros((0,), dtype=torch.int64)
            }
        else:
            targets = {
                'boxes': torch.as_tensor(boxes, dtype=torch.float32),
                'labels': torch.as_tensor(labels, dtype=torch.int64),
                'masks': torch.as_tensor(masks, dtype=torch.uint8),
                'image_id': torch.tensor([0]),
                'area': torch.as_tensor(areas, dtype=torch.float32),
                'iscrowd': torch.as_tensor(iscrowds, dtype=torch.int64)
            }
        
        return targets
    
    def _create_mask_from_segmentation(self, segmentation, width, height):
        """Create binary mask from segmentation polygon."""
        mask = np.zeros((height, width), dtype=np.uint8)
        
        for seg in segmentation:
            if len(seg) >= 6:  # At least 3 points (x,y pairs)
                # Reshape to pairs of coordinates
                poly = np.array(seg).reshape(-1, 2).astype(np.int32)
                cv2.fillPoly(mask, [poly], 1)
        
        return mask


class ECGMaskRCNNTrainer:
    """
    Trainer for ECG Mask R-CNN model.
    """
    
    def __init__(self, data_path: str, device='auto'):
        self.data_path = Path(data_path)
        
        # Setup device
        if device == 'auto':
            if torch.backends.mps.is_available():
                self.device = torch.device('mps')
            elif torch.cuda.is_available():
                self.device = torch.device('cuda')
            else:
                self.device = torch.device('cpu')
        else:
            self.device = torch.device(device)
        
        print(f"🚀 Using device: {self.device}")
        
        # Create transforms
        self.train_transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        
        self.val_transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        
        # Create datasets
        self.train_dataset = ECGWaveDataset(data_path, 'train', self.train_transform)
        self.val_dataset = ECGWaveDataset(data_path, 'val', self.val_transform)
        
        # Create data loaders
        self.train_loader = DataLoader(
            self.train_dataset, 
            batch_size=2,  # Small batch size for Mask R-CNN
            shuffle=True, 
            num_workers=0,  # macOS compatibility
            collate_fn=self._collate_fn
        )
        
        self.val_loader = DataLoader(
            self.val_dataset, 
            batch_size=2, 
            shuffle=False, 
            num_workers=0,
            collate_fn=self._collate_fn
        )
        
        # Create model
        self.model = create_ecg_maskrcnn(pretrained=True).to(self.device)
        
        # Optimizer
        params = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = optim.SGD(
            params, 
            lr=0.005, 
            momentum=0.9, 
            weight_decay=0.0005
        )
        
        # Learning rate scheduler
        self.lr_scheduler = optim.lr_scheduler.StepLR(
            self.optimizer, 
            step_size=3, 
            gamma=0.1
        )
        
        print(f"📊 Created Mask R-CNN trainer")
        print(f"Train samples: {len(self.train_dataset)}")
        print(f"Val samples: {len(self.val_dataset)}")
    
    def _collate_fn(self, batch):
        """Custom collate function for Mask R-CNN."""
        images = []
        targets = []
        
        for img, target in batch:
            images.append(img)
            targets.append(target)
        
        return images, targets
    
    def train_epoch(self):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        num_batches = 0
        
        for images, targets in tqdm(self.train_loader, desc="Training Mask R-CNN"):
            # Move to device
            images = [img.to(self.device) for img in images]
            targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]
            
            # Forward pass
            self.optimizer.zero_grad()
            loss_dict = self.model(images, targets)
            
            # Calculate total loss
            losses = sum(loss for loss in loss_dict.values())
            
            # Backward pass
            losses.backward()
            self.optimizer.step()
            
            total_loss += losses.item()
            num_batches += 1
        
        return total_loss / num_batches if num_batches > 0 else 0
    
    def validate(self):
        """Validate the model."""
        self.model.eval()
        total_loss = 0
        num_batches = 0
        
        with torch.no_grad():
            for images, targets in tqdm(self.val_loader, desc="Validating"):
                # Move to device
                images = [img.to(self.device) for img in images]
                targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]
                
                # Forward pass
                loss_dict = self.model(images, targets)
                losses = sum(loss for loss in loss_dict.values())
                
                total_loss += losses.item()
                num_batches += 1
        
        return total_loss / num_batches if num_batches > 0 else 0
    
    def train(self, epochs=10):
        """Train the Mask R-CNN model."""
        print(f"🎯 Starting Mask R-CNN training for {epochs} epochs...")
        
        best_loss = float('inf')
        
        for epoch in range(epochs):
            print(f"\n📊 Epoch {epoch+1}/{epochs}")
            
            # Train
            train_loss = self.train_epoch()
            
            # Validate
            val_loss = self.validate()
            
            # Update learning rate
            self.lr_scheduler.step()
            
            print(f"Train Loss: {train_loss:.4f}")
            print(f"Val Loss: {val_loss:.4f}")
            
            # Save best model
            if val_loss < best_loss:
                best_loss = val_loss
                self.save_model('best_maskrcnn_model.pth')
                print(f"🎉 New best model saved! Loss: {best_loss:.4f}")
        
        print(f"\n✅ Training completed! Best loss: {best_loss:.4f}")
    
    def save_model(self, filename):
        """Save the model."""
        models_dir = self.data_path / 'models'
        models_dir.mkdir(exist_ok=True)
        
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, models_dir / filename)
    
    def load_model(self, filename):
        """Load the model."""
        checkpoint = torch.load(
            self.data_path / 'models' / filename, 
            map_location=self.device
        )
        self.model.load_state_dict(checkpoint['model_state_dict'])
    
    def test_detection(self, image_path: str):
        """Test wave detection on a single image."""
        self.model.eval()
        
        # Load and preprocess image
        image = Image.open(image_path).convert('RGB')
        image_tensor = self.val_transform(image).to(self.device)
        
        # Detect waves
        with torch.no_grad():
            results = self.model.detect_waves(image_tensor)
        
        print(f"\n🔍 Detection Results for {Path(image_path).name}:")
        print(f"Total detections: {results['total_detections']}")
        
        for wave_type, detections in results['wave_detections'].items():
            print(f"{wave_type}: {len(detections)} detected")
            
            for i, detection in enumerate(detections):
                timing = detection['timing']
                print(f"  {i+1}. Time: {timing['center_time_s']:.2f}s, "
                      f"Duration: {timing['duration_ms']:.0f}ms, "
                      f"Confidence: {detection['confidence']:.3f}")
        
        # Clinical measurements
        if results['clinical_measurements']['heart_rate_bpm']:
            hr = results['clinical_measurements']['heart_rate_bpm']
            print(f"\n💓 Heart Rate: {hr:.0f} BPM")
        
        return results


def main():
    """Main training function."""
    import argparse
    
    parser = argparse.ArgumentParser(description='ECG Mask R-CNN Trainer')
    parser.add_argument('--mode', choices=['train', 'test'], default='train',
                        help='Mode: train or test')
    parser.add_argument('--data-path', default='data',
                        help='Path to data directory')
    parser.add_argument('--epochs', type=int, default=10,
                        help='Number of training epochs')
    parser.add_argument('--image', type=str,
                        help='Path to test image')
    parser.add_argument('--model', default='best_maskrcnn_model.pth',
                        help='Model file for testing')
    
    args = parser.parse_args()
    
    if args.mode == 'train':
        print("🚀 Starting Mask R-CNN Training...")
        trainer = ECGMaskRCNNTrainer(args.data_path)
        trainer.train(args.epochs)
        
    elif args.mode == 'test':
        if not args.image:
            print("Error: --image required for test mode")
            return
            
        print("🔍 Testing Mask R-CNN Detection...")
        trainer = ECGMaskRCNNTrainer(args.data_path)
        
        # Load trained model
        model_path = Path(args.data_path) / 'models' / args.model
        if model_path.exists():
            trainer.load_model(args.model)
            trainer.test_detection(args.image)
        else:
            print(f"Model {model_path} not found. Train first!")


if __name__ == "__main__":
    main()