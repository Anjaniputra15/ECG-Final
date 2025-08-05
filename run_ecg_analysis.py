#!/usr/bin/env python3
"""
ECG-LLM Real Analysis System
Train and run ECG analysis models with PTB-XL data.
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

class ECGDataset(Dataset):
    """Dataset for ECG images and labels."""
    
    def __init__(self, data_path, split='train', transform=None):
        self.data_path = Path(data_path)
        self.transform = transform
        
        # Load splits
        split_file = self.data_path / f'splits/{split}.json'
        with open(split_file, 'r') as f:
            self.record_ids = json.load(f)
        
        # Load classification labels
        labels_file = self.data_path / 'processed/ptbxl/classification_labels.csv'
        self.labels_df = pd.read_csv(labels_file)
        self.labels_df = self.labels_df.set_index('image_id')
        
        # Define label columns
        self.label_columns = [
            'Normal', 'Myocardial_Infarction', 'ST_T_Changes', 
            'Conduction_Disturbance', 'Hypertrophy', 'Atrial_Fibrillation',
            'Atrial_Flutter', 'SVT', 'VT', 'Bradycardia', 'Tachycardia'
        ]
        
        print(f"Loaded {len(self.record_ids)} {split} samples")
        
    def __len__(self):
        return len(self.record_ids)
        
    def __getitem__(self, idx):
        record_id = self.record_ids[idx]
        
        # Load image
        img_path = self.data_path / 'processed/ptbxl/images' / f'{record_id.replace("ptbxl_", "ecg_")}.png'
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
            
        # Load labels
        if record_id in self.labels_df.index:
            labels = self.labels_df.loc[record_id][self.label_columns].values.astype(float)
        else:
            labels = np.zeros(len(self.label_columns))
            
        return image, torch.FloatTensor(labels), record_id

class ECGClassifier(nn.Module):
    """ECG Classification Model using ResNet backbone."""
    
    def __init__(self, num_classes=11):
        super(ECGClassifier, self).__init__()
        
        # Use ResNet18 as backbone
        from torchvision.models import resnet18
        self.backbone = resnet18(pretrained=True)
        
        # Modify final layer for our classes
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, num_classes)
        
        # Add sigmoid for multi-label classification
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        x = self.backbone(x)
        return self.sigmoid(x)

class ECGTrainer:
    """ECG Model Trainer."""
    
    def __init__(self, data_path, device='auto'):
        self.data_path = Path(data_path)
        
        # Setup device
        if device == 'auto':
            if torch.backends.mps.is_available():
                self.device = torch.device('mps')  # Apple Silicon
            elif torch.cuda.is_available():
                self.device = torch.device('cuda')
            else:
                self.device = torch.device('cpu')
        else:
            self.device = torch.device(device)
            
        print(f"Using device: {self.device}")
        
        # Setup transforms
        self.train_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(0.5),
            transforms.RandomRotation(5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        self.val_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        # Create datasets
        self.train_dataset = ECGDataset(data_path, 'train', self.train_transform)
        self.val_dataset = ECGDataset(data_path, 'val', self.val_transform)
        
        # Create data loaders
        self.train_loader = DataLoader(
            self.train_dataset, batch_size=8, shuffle=True, num_workers=2
        )
        self.val_loader = DataLoader(
            self.val_dataset, batch_size=8, shuffle=False, num_workers=2
        )
        
        # Initialize model
        self.model = ECGClassifier(num_classes=11).to(self.device)
        
        # Setup loss and optimizer
        self.criterion = nn.BCELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=10, gamma=0.5)
        
    def train_epoch(self):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        
        for images, labels, _ in tqdm(self.train_loader, desc="Training"):
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            
        return total_loss / len(self.train_loader)
    
    def validate(self):
        """Validate the model."""
        self.model.eval()
        total_loss = 0
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for images, labels, _ in tqdm(self.val_loader, desc="Validating"):
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                total_loss += loss.item()
                
                all_predictions.append(outputs.cpu().numpy())
                all_labels.append(labels.cpu().numpy())
        
        predictions = np.vstack(all_predictions)
        labels = np.vstack(all_labels)
        
        # Calculate accuracy (threshold at 0.5)
        pred_binary = (predictions > 0.5).astype(int)
        accuracy = (pred_binary == labels).mean()
        
        return total_loss / len(self.val_loader), accuracy
    
    def train(self, epochs=20):
        """Train the model."""
        print(f"Starting training for {epochs} epochs...")
        
        best_acc = 0
        for epoch in range(epochs):
            print(f"\nEpoch {epoch+1}/{epochs}")
            
            # Train
            train_loss = self.train_epoch()
            
            # Validate
            val_loss, val_acc = self.validate()
            
            # Update learning rate
            self.scheduler.step()
            
            print(f"Train Loss: {train_loss:.4f}")
            print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
            
            # Save best model
            if val_acc > best_acc:
                best_acc = val_acc
                self.save_model('best_model.pth')
                print(f"New best model saved! Accuracy: {best_acc:.4f}")
        
        print(f"\nTraining completed! Best accuracy: {best_acc:.4f}")
        
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
        checkpoint = torch.load(self.data_path / 'models' / filename, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
    def predict(self, image_path):
        """Predict on a single ECG image."""
        self.model.eval()
        
        # Load and preprocess image
        image = Image.open(image_path).convert('RGB')
        image = self.val_transform(image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            output = self.model(image)
            predictions = output.cpu().numpy()[0]
            
        # Create results
        label_names = [
            'Normal', 'Myocardial_Infarction', 'ST_T_Changes', 
            'Conduction_Disturbance', 'Hypertrophy', 'Atrial_Fibrillation',
            'Atrial_Flutter', 'SVT', 'VT', 'Bradycardia', 'Tachycardia'
        ]
        
        results = {}
        for i, label in enumerate(label_names):
            results[label] = float(predictions[i])
            
        return results

def main():
    parser = argparse.ArgumentParser(description='ECG-LLM Analysis System')
    parser.add_argument('--mode', choices=['train', 'predict'], required=True,
                        help='Mode: train or predict')
    parser.add_argument('--data-path', default='data',
                        help='Path to data directory')
    parser.add_argument('--epochs', type=int, default=20,
                        help='Number of training epochs')
    parser.add_argument('--image', type=str,
                        help='Path to ECG image for prediction')
    parser.add_argument('--model', default='best_model.pth',
                        help='Model file for prediction')
    
    args = parser.parse_args()
    
    if args.mode == 'train':
        print("🚀 Starting ECG-LLM Training...")
        trainer = ECGTrainer(args.data_path)
        trainer.train(args.epochs)
        
    elif args.mode == 'predict':
        if not args.image:
            print("Error: --image required for prediction mode")
            return
            
        print("🔍 Running ECG Analysis...")
        trainer = ECGTrainer(args.data_path)
        
        # Load trained model
        model_path = Path(args.data_path) / 'models' / args.model
        if not model_path.exists():
            print(f"Error: Model {model_path} not found. Train first!")
            return
            
        trainer.load_model(args.model)
        
        # Predict
        results = trainer.predict(args.image)
        
        print(f"\n📊 ECG Analysis Results for: {args.image}")
        print("=" * 50)
        
        # Sort by confidence
        sorted_results = sorted(results.items(), key=lambda x: x[1], reverse=True)
        
        for condition, confidence in sorted_results:
            status = "🔴 DETECTED" if confidence > 0.5 else "⚪ Normal"
            print(f"{condition:20s}: {confidence:6.3%} {status}")

if __name__ == "__main__":
    main()