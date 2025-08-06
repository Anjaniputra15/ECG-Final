#!/usr/bin/env python3
"""
ECG-LLM ViT Analysis System
Advanced ECG analysis using Vision Transformer for medical-grade accuracy.
"""

import os
import sys
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image
import pandas as pd
import numpy as np
from pathlib import Path
import argparse
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Import our ViT model
sys.path.append(str(Path(__file__).parent))
from models.vision_transformer_ecg import ECGViTClassifier, create_ecg_vit_model

class ECGViTDataset(Dataset):
    """Enhanced dataset for ViT ECG analysis."""
    
    def __init__(self, data_path, split='train', transform=None):
        self.data_path = Path(data_path)
        self.transform = transform
        
        # Load splits from new full dataset
        split_file = self.data_path / 'processed/ptbxl_full/metadata/dataset_splits.json'
        if not split_file.exists():
            # Fallback to original dataset
            split_file = self.data_path / f'splits/{split}.json'
            with open(split_file, 'r') as f:
                self.record_ids = json.load(f)
            self.images_path = self.data_path / 'processed/ptbxl/images'
            self.labels_file = self.data_path / 'processed/ptbxl/classification_labels.csv'
        else:
            # Use new full dataset
            with open(split_file, 'r') as f:
                splits = json.load(f)
                self.record_ids = [f"ptbxl_{rid:05d}" for rid in splits[split]]
            self.images_path = self.data_path / 'processed/ptbxl_full/images'
            self.labels_file = self.data_path / 'processed/ptbxl_full/classification_labels.csv'
        
        # Load labels if available
        if self.labels_file.exists():
            self.labels_df = pd.read_csv(self.labels_file)
            if 'image_id' in self.labels_df.columns:
                self.labels_df = self.labels_df.set_index('image_id')
        else:
            self.labels_df = None
        
        # Define label columns
        self.label_columns = [
            'Normal', 'Myocardial_Infarction', 'ST_T_Changes', 
            'Conduction_Disturbance', 'Hypertrophy', 'Atrial_Fibrillation',
            'Atrial_Flutter', 'SVT', 'VT', 'Bradycardia', 'Tachycardia'
        ]
        
        print(f"Loaded {len(self.record_ids)} {split} samples from ViT dataset")
        
    def __len__(self):
        return len(self.record_ids)
        
    def __getitem__(self, idx):
        record_id = self.record_ids[idx]
        
        # Load image with multiple path attempts
        possible_paths = [
            self.images_path / f'{record_id.replace("ptbxl_", "ecg_")}.png',
            self.images_path / f'ecg_{record_id:05d}.png' if isinstance(record_id, int) else None,
            self.images_path / f'{record_id}.png'
        ]
        
        image = None
        for img_path in possible_paths:
            if img_path and img_path.exists():
                image = Image.open(img_path).convert('RGB')
                break
        
        if image is None:
            # Create dummy image if not found
            image = Image.new('RGB', (224, 224), color='white')
            
        if self.transform:
            image = self.transform(image)
            
        # Load labels if available
        if self.labels_df is not None and record_id in self.labels_df.index:
            labels = self.labels_df.loc[record_id][self.label_columns].values.astype(float)
        else:
            labels = np.zeros(len(self.label_columns))
            
        return image, torch.FloatTensor(labels), record_id

class ECGViTTrainer:
    """Advanced ECG ViT Trainer with medical optimizations."""
    
    def __init__(self, data_path, model_type='base', device='auto'):
        self.data_path = Path(data_path)
        
        # Setup device with Apple Silicon support
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
        
        # Enhanced transforms for ViT
        self.train_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(0.3),  # Reduced for medical data
            transforms.RandomRotation(3),  # Minimal rotation for ECGs
            transforms.ColorJitter(brightness=0.1, contrast=0.1),  # Subtle augmentation
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])  # ImageNet normalization
        ])
        
        self.val_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        # Create datasets
        self.train_dataset = ECGViTDataset(data_path, 'train', self.train_transform)
        self.val_dataset = ECGViTDataset(data_path, 'val', self.val_transform)
        
        # Create data loaders with macOS compatibility
        self.train_loader = DataLoader(
            self.train_dataset, batch_size=16, shuffle=True, 
            num_workers=0, pin_memory=False  # Fixed for macOS
        )
        self.val_loader = DataLoader(
            self.val_dataset, batch_size=16, shuffle=False, 
            num_workers=0, pin_memory=False  # Fixed for macOS
        )
        
        # Initialize ViT model
        self.model = create_ecg_vit_model(
            num_classes=11, 
            model_type=model_type,
            ensemble=False
        ).to(self.device)
        
        print(f"📊 Created ViT-{model_type.upper()} model with {sum(p.numel() for p in self.model.parameters()):,} parameters")
        
        # Medical-optimized training setup
        self.criterion = nn.BCELoss()
        self.optimizer = optim.AdamW(
            self.model.parameters(), 
            lr=3e-4,  # Lower learning rate for ViT
            weight_decay=0.05  # Stronger regularization
        )
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=50, eta_min=1e-6
        )
        
    def train_epoch(self):
        """Train for one epoch with ViT optimizations."""
        self.model.train()
        total_loss = 0
        
        for images, labels, _ in tqdm(self.train_loader, desc="Training ViT"):
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            self.optimizer.zero_grad()
            
            # Forward pass
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            
            # Backward pass with gradient clipping for stability
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
            
        return total_loss / len(self.train_loader)
    
    def validate(self):
        """Validate with medical metrics."""
        self.model.eval()
        total_loss = 0
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for images, labels, _ in tqdm(self.val_loader, desc="Validating ViT"):
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                total_loss += loss.item()
                
                all_predictions.append(outputs.cpu().numpy())
                all_labels.append(labels.cpu().numpy())
        
        predictions = np.vstack(all_predictions)
        labels = np.vstack(all_labels)
        
        # Calculate medical-grade metrics
        pred_binary = (predictions > 0.5).astype(int)
        accuracy = (pred_binary == labels).mean()
        
        # Calculate per-condition accuracy
        condition_accuracies = []
        condition_names = [
            'Normal', 'MI', 'ST-T', 'Conduction', 'Hypertrophy', 
            'AFib', 'AFlutter', 'SVT', 'VT', 'Brady', 'Tachy'
        ]
        
        for i, condition in enumerate(condition_names):
            if labels[:, i].sum() > 0:  # Only if condition exists in validation
                cond_acc = (pred_binary[:, i] == labels[:, i]).mean()
                condition_accuracies.append((condition, cond_acc))
        
        return total_loss / len(self.val_loader), accuracy, condition_accuracies
    
    def train(self, epochs=30):
        """Train ViT model with medical optimizations."""
        print(f"🎯 Starting ViT training for {epochs} epochs...")
        
        best_acc = 0
        for epoch in range(epochs):
            print(f"\n📊 Epoch {epoch+1}/{epochs}")
            
            # Train
            train_loss = self.train_epoch()
            
            # Validate
            val_loss, val_acc, condition_accs = self.validate()
            
            # Update learning rate
            self.scheduler.step()
            
            print(f"Train Loss: {train_loss:.4f}")
            print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
            
            # Print condition-specific accuracies
            if condition_accs:
                print("Condition Accuracies:")
                for condition, acc in condition_accs:
                    print(f"  {condition}: {acc:.3f}")
            
            # Save best model
            if val_acc > best_acc:
                best_acc = val_acc
                self.save_model('best_vit_model.pth')
                print(f"🎉 New best ViT model saved! Accuracy: {best_acc:.4f}")
        
        print(f"\n✅ ViT Training completed! Best accuracy: {best_acc:.4f}")
        
    def save_model(self, filename):
        """Save ViT model."""
        models_dir = self.data_path / 'models'
        models_dir.mkdir(exist_ok=True)
        
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'model_type': 'vision_transformer'
        }, models_dir / filename)
        
    def load_model(self, filename):
        """Load ViT model."""
        checkpoint = torch.load(self.data_path / 'models' / filename, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
    def predict_with_report(self, image_path):
        """Generate comprehensive medical report using ViT."""
        self.model.eval()
        
        # Load and preprocess image
        image = Image.open(image_path).convert('RGB')
        image = self.val_transform(image).unsqueeze(0).to(self.device)
        
        # Generate medical report
        report = self.model.get_medical_report(image, patient_id=Path(image_path).stem)
        
        return report

def main():
    parser = argparse.ArgumentParser(description='ECG-LLM ViT Analysis System')
    parser.add_argument('--mode', choices=['train', 'predict', 'report'], required=True,
                        help='Mode: train, predict, or report')
    parser.add_argument('--data-path', default='data',
                        help='Path to data directory')
    parser.add_argument('--epochs', type=int, default=30,
                        help='Number of training epochs')
    parser.add_argument('--image', type=str,
                        help='Path to ECG image for prediction')
    parser.add_argument('--model', default='best_vit_model.pth',
                        help='Model file for prediction')
    parser.add_argument('--model-type', default='base', choices=['base', 'large', 'huge'],
                        help='ViT model size')
    
    args = parser.parse_args()
    
    if args.mode == 'train':
        print("🚀 Starting ECG-ViT Training...")
        trainer = ECGViTTrainer(args.data_path, model_type=args.model_type)
        trainer.train(args.epochs)
        
    elif args.mode in ['predict', 'report', 'ensemble']:
        if not args.image:
            print("Error: --image required for prediction/report mode")
            return
            
        print("🔍 Running ECG-ViT Analysis...")
        trainer = ECGViTTrainer(args.data_path, model_type=args.model_type)
        
        # Load trained model
        model_path = Path(args.data_path) / 'models' / args.model
        if not model_path.exists():
            print(f"Error: Model {model_path} not found. Train first!")
            return
            
        trainer.load_model(args.model)
        
        if args.mode == 'report':
            # Generate comprehensive medical report
            report = trainer.predict_with_report(args.image)
            
            print(f"\n🏥 COMPREHENSIVE ECG MEDICAL REPORT")
            print("=" * 60)
            print(f"Patient ID: {report['patient_id']}")
            print(f"Analysis: {report['analysis_type']}")
            print(f"Model: {report['model_version']}")
            print(f"Overall Risk: {report['overall_risk']}")
            
            if report['requires_urgent_attention']:
                print("🚨 URGENT ATTENTION REQUIRED 🚨")
            
            print(f"\n📋 CLINICAL FINDINGS:")
            for finding in report['findings']:
                status_icon = "🔴" if finding['status'] == 'DETECTED' else "✅"
                print(f"{status_icon} {finding['condition']}: {finding['confidence']:.1%}")
                print(f"   {finding['clinical_significance']}")
            
            print(f"\n💡 RECOMMENDATIONS:")
            for i, rec in enumerate(report['recommendations'], 1):
                print(f"{i}. {rec}")
                
        else:
            # Simple prediction mode
            trainer.model.eval()
            image = Image.open(args.image).convert('RGB')
            image = trainer.val_transform(image).unsqueeze(0).to(trainer.device)
            
            with torch.no_grad():
                output = trainer.model(image)
                predictions = output.cpu().numpy()[0]
            
            condition_names = [
                'Normal', 'Myocardial_Infarction', 'ST_T_Changes',
                'Conduction_Disturbance', 'Hypertrophy', 'Atrial_Fibrillation',
                'Atrial_Flutter', 'SVT', 'VT', 'Bradycardia', 'Tachycardia'
            ]
            
            print(f"\n📊 ECG-ViT Analysis Results for: {args.image}")
            print("=" * 50)
            
            # Sort by confidence
            results = list(zip(condition_names, predictions))
            results.sort(key=lambda x: x[1], reverse=True)
            
            for condition, confidence in results:
                status = " DETECTED" if confidence > 0.5 else " Normal"
                print(f"{condition:20s}: {confidence:6.3%} {status}")

if __name__ == "__main__":
    main()