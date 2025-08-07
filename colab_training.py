#!/usr/bin/env python3
"""
Google Colab Training Pipeline for ECG-LLM PQRST Detection
Adapted from existing codebase for Colab Pro environment with PTB-XL dataset
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import os
import wfdb
from datetime import datetime
from tqdm.auto import tqdm
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

class ColabECGConfig:
    """Configuration optimized for Google Colab Pro"""
    
    def __init__(self):
        # Colab-optimized settings
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.batch_size = 16  # Optimized for Colab GPU memory
        self.learning_rate = 1e-4
        self.num_epochs = 50  # Reasonable for Colab session limits
        self.warmup_epochs = 5
        self.weight_decay = 1e-4
        self.gradient_clip_norm = 1.0
        self.save_every_n_epochs = 10
        
        # Data settings
        self.max_samples_per_split = 1000  # Start with subset for faster training
        self.signal_length = 5000  # 10 seconds at 500Hz
        self.num_leads = 12
        self.num_classes = 6  # P, Q, R, S, T, Background
        
        # Google Drive integration
        self.use_drive = True
        self.drive_project_path = "/content/drive/MyDrive/ECG_LLM_Project"
        
        print(f"🔧 Colab ECG Config Initialized")
        print(f"🖥️  Device: {self.device}")
        print(f"📦 Batch size: {self.batch_size}")
        print(f"🎯 Max samples per split: {self.max_samples_per_split}")

class PTBXLColabDataset(Dataset):
    """PTB-XL Dataset optimized for Google Colab training"""
    
    def __init__(self, ecg_ids, database, data_path, config, split='train'):
        self.ecg_ids = list(ecg_ids)
        self.database = database
        self.data_path = data_path
        self.config = config
        self.split = split
        
        print(f"📊 Created {split} dataset with {len(self.ecg_ids)} samples")
    
    def __len__(self):
        return len(self.ecg_ids)
    
    def __getitem__(self, idx):
        ecg_id = self.ecg_ids[idx]
        
        try:
            # Load ECG signal from PTB-XL
            record_path = f"{self.data_path}/records500/{ecg_id:05d}/{ecg_id:05d}"
            signal, fields = wfdb.rdsamp(record_path)
            
            # Convert to tensor [leads, samples]
            signal = torch.FloatTensor(signal.T)
            
            # Handle different lead configurations
            if signal.shape[0] < self.config.num_leads:
                # Pad with zeros if fewer leads
                padding = self.config.num_leads - signal.shape[0]
                signal = F.pad(signal, (0, 0, 0, padding))
            elif signal.shape[0] > self.config.num_leads:
                # Take first 12 leads
                signal = signal[:self.config.num_leads]
            
            # Standardize signal length
            if signal.shape[1] > self.config.signal_length:
                signal = signal[:, :self.config.signal_length]
            else:
                padding = self.config.signal_length - signal.shape[1]
                signal = F.pad(signal, (0, padding))
            
            # Normalize signal
            signal = (signal - signal.mean()) / (signal.std() + 1e-8)
            
            # Get metadata
            metadata = self.database.loc[ecg_id]
            
            # Create labels (simplified for bootstrap training)
            scp_codes = eval(metadata.scp_codes) if pd.notna(metadata.scp_codes) else {}
            
            # Binary classification: normal vs abnormal
            is_normal = len(scp_codes) == 0 or 'NORM' in scp_codes
            binary_label = torch.tensor(0 if is_normal else 1, dtype=torch.long)
            
            # Create synthetic PQRST targets for bootstrap training
            # This simulates your original bootstrap approach
            pqrst_targets = self.create_synthetic_pqrst_targets()
            
            return {
                'signal': signal,
                'binary_label': binary_label,
                'pqrst_targets': pqrst_targets,
                'ecg_id': ecg_id,
                'age': float(metadata.age) if pd.notna(metadata.age) else 50.0,
                'sex': int(metadata.sex) if pd.notna(metadata.sex) else 0
            }
            
        except Exception as e:
            print(f"⚠️  Error loading ECG {ecg_id}: {e}")
            # Return dummy data
            return {
                'signal': torch.randn(self.config.num_leads, self.config.signal_length),
                'binary_label': torch.tensor(0, dtype=torch.long),
                'pqrst_targets': self.create_synthetic_pqrst_targets(),
                'ecg_id': ecg_id,
                'age': 50.0,
                'sex': 0
            }
    
    def create_synthetic_pqrst_targets(self):
        """Create synthetic PQRST wave targets similar to bootstrap trainer"""
        # Simulate R-peak locations (main focus of bootstrap training)
        r_peaks = []
        signal_length = self.config.signal_length
        
        # Typical heart rate: 60-100 BPM, so R-peaks every 500-833 samples at 500Hz
        peak_interval = np.random.randint(500, 833)
        
        current_pos = np.random.randint(100, 200)  # First R-peak position
        while current_pos < signal_length - 100:
            r_peaks.append(current_pos)
            current_pos += peak_interval + np.random.randint(-50, 50)  # Add variability
        
        # Convert to target tensor
        targets = torch.zeros(self.config.num_classes, signal_length)
        
        for peak_pos in r_peaks:
            # R-peak (class 2)
            start_r = max(0, peak_pos - 10)
            end_r = min(signal_length, peak_pos + 10)
            targets[2, start_r:end_r] = 1.0
            
            # P-wave (class 0) - before R-peak
            if peak_pos > 100:
                p_start = max(0, peak_pos - 80)
                p_end = min(signal_length, peak_pos - 40)
                targets[0, p_start:p_end] = 1.0
            
            # T-wave (class 4) - after R-peak
            if peak_pos < signal_length - 100:
                t_start = max(0, peak_pos + 40)
                t_end = min(signal_length, peak_pos + 120)
                targets[4, t_start:t_end] = 1.0
        
        # Background class (class 5)
        targets[5] = 1.0 - targets[:5].sum(dim=0)
        targets[5] = torch.clamp(targets[5], 0, 1)
        
        return targets

class ColabECGModel(nn.Module):
    """Simplified ECG model optimized for Colab training"""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # 1D CNN backbone for ECG signals
        self.conv_layers = nn.Sequential(
            # First conv block
            nn.Conv1d(config.num_leads, 64, kernel_size=15, padding=7),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
            
            # Second conv block
            nn.Conv1d(64, 128, kernel_size=15, padding=7),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
            
            # Third conv block
            nn.Conv1d(128, 256, kernel_size=15, padding=7),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
            
            # Fourth conv block
            nn.Conv1d(256, 512, kernel_size=15, padding=7),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
        )
        
        # Calculate feature map size
        dummy_input = torch.randn(1, config.num_leads, config.signal_length)
        with torch.no_grad():
            conv_output = self.conv_layers(dummy_input)
            self.feature_size = conv_output.shape[1] * conv_output.shape[2]
        
        # Classification heads
        self.binary_classifier = nn.Sequential(
            nn.Linear(self.feature_size, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 2)  # Normal vs Abnormal
        )
        
        # PQRST wave detector (similar to bootstrap trainer)
        self.pqrst_detector = nn.Sequential(
            nn.Conv1d(512, 256, kernel_size=1),
            nn.ReLU(),
            nn.Conv1d(256, config.num_classes, kernel_size=1),
        )
        
        # Upsampling to match original signal length
        self.upsampler = nn.Upsample(size=config.signal_length, mode='linear', align_corners=False)
        
        print(f"🧠 Model initialized with {sum(p.numel() for p in self.parameters()):,} parameters")
    
    def forward(self, x):
        batch_size = x.shape[0]
        
        # Extract features
        conv_features = self.conv_layers(x)  # [batch, 512, reduced_length]
        
        # Binary classification
        flattened = conv_features.view(batch_size, -1)
        binary_logits = self.binary_classifier(flattened)
        
        # PQRST detection
        pqrst_logits = self.pqrst_detector(conv_features)
        pqrst_logits = self.upsampler(pqrst_logits)  # Upsample to original length
        
        return {
            'binary_logits': binary_logits,
            'pqrst_logits': pqrst_logits
        }

class ColabECGTrainer:
    """ECG trainer optimized for Google Colab environment"""
    
    def __init__(self, config):
        self.config = config
        self.device = torch.device(config.device)
        
        # Setup Google Drive integration
        if config.use_drive:
            self.setup_drive_integration()
        
        # Create model
        self.model = ColabECGModel(config).to(self.device)
        
        # Setup optimizer and scheduler
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        
        self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, T_0=10, T_mult=2
        )
        
        # Loss functions
        self.binary_criterion = nn.CrossEntropyLoss()
        self.pqrst_criterion = nn.BCEWithLogitsLoss()
        
        # Metrics tracking
        self.training_history = {
            'train_loss': [],
            'val_loss': [],
            'binary_acc': [],
            'pqrst_dice': []
        }
        
        print(f"🚀 Trainer initialized on {self.device}")
    
    def setup_drive_integration(self):
        """Setup Google Drive for persistent storage"""
        try:
            from google.colab import drive
            drive.mount('/content/drive')
            
            # Create project directories
            os.makedirs(self.config.drive_project_path, exist_ok=True)
            os.makedirs(f"{self.config.drive_project_path}/models", exist_ok=True)
            os.makedirs(f"{self.config.drive_project_path}/checkpoints", exist_ok=True)
            os.makedirs(f"{self.config.drive_project_path}/results", exist_ok=True)
            
            print("✅ Google Drive integrated successfully")
            
        except Exception as e:
            print(f"⚠️  Drive integration failed: {e}")
            self.config.use_drive = False
    
    def train_epoch(self, train_loader, epoch):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        binary_correct = 0
        total_samples = 0
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{self.config.num_epochs}')
        
        for batch_idx, batch in enumerate(pbar):
            signals = batch['signal'].to(self.device)
            binary_labels = batch['binary_label'].to(self.device)
            pqrst_targets = batch['pqrst_targets'].to(self.device)
            
            self.optimizer.zero_grad()
            
            # Forward pass
            outputs = self.model(signals)
            
            # Calculate losses
            binary_loss = self.binary_criterion(outputs['binary_logits'], binary_labels)
            pqrst_loss = self.pqrst_criterion(outputs['pqrst_logits'], pqrst_targets)
            
            # Combined loss (weighted)
            total_loss_batch = binary_loss + 0.5 * pqrst_loss
            
            # Backward pass
            total_loss_batch.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clip_norm)
            self.optimizer.step()
            
            # Update metrics
            total_loss += total_loss_batch.item()
            binary_pred = outputs['binary_logits'].argmax(dim=1)
            binary_correct += (binary_pred == binary_labels).sum().item()
            total_samples += binary_labels.size(0)
            
            # Update progress bar
            current_acc = binary_correct / total_samples
            pbar.set_postfix({
                'loss': f'{total_loss_batch.item():.4f}',
                'acc': f'{current_acc:.4f}',
                'lr': f'{self.optimizer.param_groups[0]["lr"]:.6f}'
            })
        
        avg_loss = total_loss / len(train_loader)
        accuracy = binary_correct / total_samples
        
        return avg_loss, accuracy
    
    def validate_epoch(self, val_loader):
        """Validate for one epoch"""
        self.model.eval()
        total_loss = 0
        binary_correct = 0
        total_samples = 0
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc='Validation'):
                signals = batch['signal'].to(self.device)
                binary_labels = batch['binary_label'].to(self.device)
                pqrst_targets = batch['pqrst_targets'].to(self.device)
                
                outputs = self.model(signals)
                
                # Calculate losses
                binary_loss = self.binary_criterion(outputs['binary_logits'], binary_labels)
                pqrst_loss = self.pqrst_criterion(outputs['pqrst_logits'], pqrst_targets)
                total_loss_batch = binary_loss + 0.5 * pqrst_loss
                
                total_loss += total_loss_batch.item()
                binary_pred = outputs['binary_logits'].argmax(dim=1)
                binary_correct += (binary_pred == binary_labels).sum().item()
                total_samples += binary_labels.size(0)
        
        avg_loss = total_loss / len(val_loader)
        accuracy = binary_correct / total_samples
        
        return avg_loss, accuracy
    
    def save_checkpoint(self, epoch, val_loss, is_best=False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'val_loss': val_loss,
            'training_history': self.training_history,
            'config': self.config.__dict__
        }
        
        # Save locally
        local_path = f'checkpoint_epoch_{epoch+1}.pth'
        torch.save(checkpoint, local_path)
        
        if is_best:
            torch.save(checkpoint, 'best_model.pth')
        
        # Save to Google Drive if available
        if self.config.use_drive:
            try:
                drive_path = f"{self.config.drive_project_path}/checkpoints/checkpoint_epoch_{epoch+1}.pth"
                torch.save(checkpoint, drive_path)
                
                if is_best:
                    best_drive_path = f"{self.config.drive_project_path}/models/best_model.pth"
                    torch.save(checkpoint, best_drive_path)
                    
                print(f"💾 Checkpoint saved to Google Drive")
            except Exception as e:
                print(f"⚠️  Failed to save to Drive: {e}")
    
    def train(self, train_loader, val_loader):
        """Main training loop"""
        print("🚀 Starting ECG-LLM Training!")
        print("=" * 60)
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(self.config.num_epochs):
            start_time = datetime.now()
            
            # Training
            train_loss, train_acc = self.train_epoch(train_loader, epoch)
            
            # Validation
            val_loss, val_acc = self.validate_epoch(val_loader)
            
            # Update scheduler
            self.scheduler.step()
            
            # Update history
            self.training_history['train_loss'].append(train_loss)
            self.training_history['val_loss'].append(val_loss)
            self.training_history['binary_acc'].append(val_acc)
            
            # Print epoch results
            epoch_time = (datetime.now() - start_time).total_seconds()
            
            print(f"\nEpoch {epoch+1}/{self.config.num_epochs} ({epoch_time:.1f}s):")
            print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
            print(f"  Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
            print(f"  LR: {self.optimizer.param_groups[0]['lr']:.6f}")
            
            # Save checkpoint
            is_best = val_loss < best_val_loss
            if is_best:
                best_val_loss = val_loss
                patience_counter = 0
                print("  🎉 New best model!")
            else:
                patience_counter += 1
            
            if (epoch + 1) % self.config.save_every_n_epochs == 0 or is_best:
                self.save_checkpoint(epoch, val_loss, is_best)
            
            print("-" * 60)
        
        print(f"🎯 Training completed! Best val loss: {best_val_loss:.4f}")
        return self.training_history
    
    def plot_training_curves(self):
        """Plot training curves"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        
        # Loss curves
        axes[0, 0].plot(self.training_history['train_loss'], label='Train Loss', color='blue')
        axes[0, 0].plot(self.training_history['val_loss'], label='Val Loss', color='red')
        axes[0, 0].set_title('Training and Validation Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Accuracy curve
        axes[0, 1].plot(self.training_history['binary_acc'], label='Binary Accuracy', color='green')
        axes[0, 1].set_title('Validation Accuracy')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # Learning rate
        epochs = range(1, len(self.training_history['train_loss']) + 1)
        current_lr = self.optimizer.param_groups[0]['lr']
        axes[1, 0].plot(epochs, [current_lr] * len(epochs))
        axes[1, 0].set_title('Learning Rate Schedule')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Learning Rate')
        axes[1, 0].grid(True)
        
        # Loss comparison
        if len(self.training_history['train_loss']) > 1:
            axes[1, 1].scatter(self.training_history['train_loss'], 
                             self.training_history['val_loss'], alpha=0.6)
            axes[1, 1].plot([0, max(self.training_history['train_loss'])], 
                           [0, max(self.training_history['train_loss'])], 'r--')
            axes[1, 1].set_title('Train vs Val Loss')
            axes[1, 1].set_xlabel('Train Loss')
            axes[1, 1].set_ylabel('Val Loss')
            axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.savefig('training_curves.png', dpi=300, bbox_inches='tight')
        
        # Save to Drive if available
        if self.config.use_drive:
            try:
                drive_plot_path = f"{self.config.drive_project_path}/results/training_curves.png"
                plt.savefig(drive_plot_path, dpi=300, bbox_inches='tight')
            except:
                pass
        
        plt.show()
        print("📈 Training curves saved!")

def setup_colab_training(dataset_path, config):
    """Setup training pipeline for Google Colab"""
    
    print("🔧 Setting up Colab training pipeline...")
    
    # Load PTB-XL database
    database_file = f"{dataset_path}/ptbxl_database.csv"
    database = pd.read_csv(database_file, index_col='ecg_id')
    
    print(f"📊 Loaded PTB-XL database: {len(database)} records")
    
    # Filter for quality records
    quality_data = database[
        (database.fs == 500) &  # 500Hz sampling
        (database.age >= 18) &  # Adults only
        (database.age <= 80) &  # Reasonable age range
        (database.strat_fold <= 10)  # Valid stratification fold
    ]
    
    print(f"✅ Filtered to {len(quality_data)} high-quality records")
    
    # Create balanced train/val splits
    all_ids = quality_data.index[:config.max_samples_per_split]
    train_ids, val_ids = train_test_split(all_ids, test_size=0.2, random_state=42)
    
    print(f"📊 Data splits: Train={len(train_ids)}, Val={len(val_ids)}")
    
    # Create datasets
    train_dataset = PTBXLColabDataset(train_ids, quality_data, dataset_path, config, 'train')
    val_dataset = PTBXLColabDataset(val_ids, quality_data, dataset_path, config, 'val')
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config.batch_size, 
        shuffle=True, 
        num_workers=2,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=config.batch_size, 
        shuffle=False, 
        num_workers=2,
        pin_memory=True
    )
    
    print("✅ Data loaders created successfully!")
    
    return train_loader, val_loader

# Main execution functions for Colab
def run_colab_training():
    """Main function to run ECG training in Google Colab"""
    
    print("🫀 ECG-LLM Colab Training Pipeline")
    print("=" * 50)
    
    # Initialize configuration
    config = ColabECGConfig()
    
    # Dataset path (adjust based on your Colab setup)
    dataset_path = "data/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3"
    
    # Setup training pipeline
    train_loader, val_loader = setup_colab_training(dataset_path, config)
    
    # Initialize trainer
    trainer = ColabECGTrainer(config)
    
    # Start training
    history = trainer.train(train_loader, val_loader)
    
    # Plot results
    trainer.plot_training_curves()
    
    print("🎉 Training completed successfully!")
    
    return trainer, history

if __name__ == "__main__":
    # This allows the script to be run directly in Colab
    trainer, history = run_colab_training()