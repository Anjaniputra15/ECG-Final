#!/usr/bin/env python3
"""
Advanced Training Pipeline for ECG PQRST Detection Models
Supports all 4 approaches: ViT-DETR, Multi-Modal, HuBERT-ECG, Mask R-CNN
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Union, Any
import numpy as np
from pathlib import Path
import json
import logging
from tqdm import tqdm
#import wandb
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns

# Import all model architectures
import sys
sys.path.append(str(Path(__file__).parent.parent))

from models.backbones.vision_transformer_ecg import create_ecg_vit_detr_model
from models.backbones.multimodal_ecg import create_multimodal_ecg_model
from models.backbones.hubert_ecg import create_hubert_ecg_model
from models.backbones.maskrcnn_ecg import create_ecg_maskrcnn_model
from models.ecg_ensemble import create_ecg_ensemble

class ECGTrainingConfig:
    """Configuration for ECG model training"""
    
    def __init__(self,
                 model_type: str = "ensemble",
                 batch_size: int = 8,
                 learning_rate: float = 1e-4,
                 num_epochs: int = 100,
                 warmup_epochs: int = 5,
                 device: str = "mps",
                 mixed_precision: bool = True,
                 gradient_clip_norm: float = 1.0,
                 weight_decay: float = 1e-4,
                 scheduler_type: str = "cosine",
                 early_stopping_patience: int = 15,
                 save_every_n_epochs: int = 5,
                 validate_every_n_epochs: int = 1,
                 use_wandb: bool = True,
                 wandb_project: str = "ecg-pqrst-detection"):
        
        self.model_type = model_type
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.warmup_epochs = warmup_epochs
        self.device = device
        self.mixed_precision = mixed_precision
        self.gradient_clip_norm = gradient_clip_norm
        self.weight_decay = weight_decay
        self.scheduler_type = scheduler_type
        self.early_stopping_patience = early_stopping_patience
        self.save_every_n_epochs = save_every_n_epochs
        self.validate_every_n_epochs = validate_every_n_epochs
        self.use_wandb = use_wandb
        self.wandb_project = wandb_project
        
        # Model-specific configurations
        self.model_configs = {
            "vit_detr": {
                "learning_rate": 1e-4,
                "batch_size": 4,
                "warmup_epochs": 10,
                "weight_decay": 1e-4
            },
            "multimodal": {
                "learning_rate": 5e-5,
                "batch_size": 8,
                "warmup_epochs": 5,
                "weight_decay": 1e-5
            },
            "hubert": {
                "learning_rate": 2e-5,
                "batch_size": 6,
                "warmup_epochs": 8,
                "weight_decay": 1e-5
            },
            "maskrcnn": {
                "learning_rate": 1e-3,
                "batch_size": 4,
                "warmup_epochs": 5,
                "weight_decay": 1e-4
            },
            "ensemble": {
                "learning_rate": 1e-4,
                "batch_size": 4,
                "warmup_epochs": 10,
                "weight_decay": 1e-5
            }
        }
        
        # Apply model-specific config
        if model_type in self.model_configs:
            config = self.model_configs[model_type]
            self.learning_rate = config["learning_rate"]
            self.batch_size = config["batch_size"]
            self.warmup_epochs = config["warmup_epochs"]
            self.weight_decay = config["weight_decay"]

class ECGDataset(Dataset):
    """ECG Dataset for training all model types"""
    
    def __init__(self, 
                 data_dir: Path,
                 split: str = "train",
                 image_size: Tuple[int, int] = (512, 512),
                 augment: bool = True):
        
        self.data_dir = Path(data_dir)
        self.split = split
        self.image_size = image_size
        self.augment = augment
        
        # Load metadata
        metadata_path = self.data_dir / "metadata.csv"
        self.samples = self._load_metadata(metadata_path)
        
        # Filter by split
        self.samples = [s for s in self.samples if s['split'] == split]
        
        print(f"Loaded {len(self.samples)} samples for {split} split")
    
    def _load_metadata(self, metadata_path: Path) -> List[Dict]:
        """Load metadata from CSV file"""
        samples = []
        
        if not metadata_path.exists():
            raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
        
        with open(metadata_path, 'r') as f:
            lines = f.readlines()[1:]  # Skip header
            
            for line in lines:
                parts = line.strip().split(',')
                if len(parts) >= 3:
                    sample = {
                        'image_path': self.data_dir / parts[0],
                        'mask_path': self.data_dir / parts[1], 
                        'split': parts[2]
                    }
                    samples.append(sample)
        
        return samples
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]
        
        # Load image and mask
        image = self._load_image(sample['image_path'])
        mask = self._load_mask(sample['mask_path'])
        
        # Generate synthetic PQRST points for training
        # In practice, these would come from annotations
        pqrst_points = self._generate_synthetic_pqrst_points(mask)
        
        # Apply augmentations
        if self.augment and self.split == 'train':
            image, mask, pqrst_points = self._apply_augmentations(image, mask, pqrst_points)
        
        return {
            'image': image,
            'mask': mask,
            'pqrst_points': pqrst_points,
            'classes': torch.tensor([1, 1, 1, 1, 1], dtype=torch.long),  # All points present
            'confidence': torch.tensor([0.9, 0.95, 1.0, 0.95, 0.9]),  # R-peak highest confidence
            'sample_id': idx
        }
    
    def _load_image(self, image_path: Path) -> torch.Tensor:
        """Load and preprocess ECG image"""
        try:
            import cv2
            image = cv2.imread(str(image_path))
            if image is None:
                # Create dummy image if file doesn't exist
                image = np.ones((512, 512, 3), dtype=np.uint8) * 255
            else:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image = cv2.resize(image, self.image_size)
            
            # Convert to tensor and normalize
            image = torch.from_numpy(image).float().permute(2, 0, 1) / 255.0
            
            # Normalize with ImageNet stats
            mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
            image = (image - mean) / std
            
            return image
            
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            # Return dummy image
            return torch.randn(3, *self.image_size)
    
    def _load_mask(self, mask_path: Path) -> torch.Tensor:
        """Load ECG mask"""
        try:
            import cv2
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if mask is None:
                # Create dummy mask
                mask = np.zeros(self.image_size, dtype=np.uint8)
                # Add some ECG-like traces
                h, w = self.image_size
                mask[h//2-10:h//2+10, :] = 255
            else:
                mask = cv2.resize(mask, self.image_size)
            
            # Convert to tensor and normalize
            mask = torch.from_numpy(mask).float() / 255.0
            
            return mask
            
        except Exception as e:
            print(f"Error loading mask {mask_path}: {e}")
            # Return dummy mask
            return torch.zeros(*self.image_size)
    
    def _generate_synthetic_pqrst_points(self, mask: torch.Tensor) -> torch.Tensor:
        """Generate synthetic PQRST points based on mask"""
        h, w = mask.shape
        
        # Find mask center line
        mask_indices = torch.nonzero(mask > 0.5)
        if len(mask_indices) == 0:
            # Default positions if no mask
            points = torch.tensor([
                [w * 0.2, h * 0.5],  # P
                [w * 0.35, h * 0.5], # Q  
                [w * 0.5, h * 0.5],  # R
                [w * 0.65, h * 0.5], # S
                [w * 0.8, h * 0.5]   # T
            ])
        else:
            # Generate points along mask
            center_y = mask_indices[:, 0].float().mean()
            x_positions = [w * pos for pos in [0.2, 0.35, 0.5, 0.65, 0.8]]
            
            points = []
            for x_pos in x_positions:
                # Add some noise for realism
                y_noise = torch.randn(1) * 10
                points.append([x_pos, center_y + y_noise])
            
            points = torch.tensor(points)
        
        # Normalize to [0, 1]
        points[:, 0] /= w
        points[:, 1] /= h
        
        # Clamp to valid range
        points = torch.clamp(points, 0, 1)
        
        return points
    
    def _apply_augmentations(self, image: torch.Tensor, mask: torch.Tensor, 
                           points: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Apply data augmentations"""
        # Horizontal flip
        if torch.rand(1) < 0.5:
            image = torch.flip(image, dims=[2])
            mask = torch.flip(mask, dims=[1])
            points[:, 0] = 1.0 - points[:, 0]  # Flip x coordinates
        
        # Add noise
        if torch.rand(1) < 0.3:
            noise = torch.randn_like(image) * 0.05
            image = image + noise
        
        # Brightness adjustment
        if torch.rand(1) < 0.3:
            brightness_factor = 0.8 + torch.rand(1) * 0.4  # 0.8 to 1.2
            image = image * brightness_factor
        
        # Clamp image values
        image = torch.clamp(image, -3, 3)  # Reasonable range for normalized images
        
        return image, mask, points

class ECGAdvancedTrainer:
    """Advanced trainer for ECG PQRST detection models"""
    
    def __init__(self, 
                 config: ECGTrainingConfig,
                 save_dir: Path,
                 data_dir: Path):
        
        self.config = config
        self.save_dir = Path(save_dir)
        self.data_dir = Path(data_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        self._setup_logging()
        
        # Setup device
        self.device = self._setup_device()
        
        # Initialize wandb
        if config.use_wandb:
            self._setup_wandb()
        
        # Training state
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        self.training_history = defaultdict(list)
        
        # Mixed precision
        self.scaler = torch.cuda.amp.GradScaler() if config.mixed_precision else None
    
    def _setup_logging(self):
        """Setup logging configuration"""
        log_file = self.save_dir / "training.log"
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def _setup_device(self) -> torch.device:
        """Setup compute device"""
        if self.config.device == "mps" and torch.backends.mps.is_available():
            device = torch.device("mps")
            self.logger.info("Using MPS (Apple Silicon) acceleration")
        elif self.config.device == "cuda" and torch.cuda.is_available():
            device = torch.device("cuda")
            self.logger.info(f"Using CUDA GPU: {torch.cuda.get_device_name()}")
        else:
            device = torch.device("cpu")
            self.logger.info("Using CPU")
        
        return device
    
    def _setup_wandb(self):
        """Setup Weights & Biases logging"""
        try:
            wandb.init(
                project=self.config.wandb_project,
                config=vars(self.config),
                name=f"ecg-{self.config.model_type}-{self.current_epoch}",
                save_code=True
            )
            self.logger.info("Weights & Biases logging initialized")
        except Exception as e:
            self.logger.warning(f"Failed to initialize wandb: {e}")
            self.config.use_wandb = False
    
    def create_model(self) -> nn.Module:
        """Create model based on configuration"""
        self.logger.info(f"Creating {self.config.model_type} model...")
        
        if self.config.model_type == "vit_detr":
            model = create_ecg_vit_detr_model(pretrained=True)
        elif self.config.model_type == "multimodal":
            model = create_multimodal_ecg_model(fusion_method="attention")
        elif self.config.model_type == "hubert":
            model = create_hubert_ecg_model(model_size="base")
        elif self.config.model_type == "maskrcnn":
            model = create_ecg_maskrcnn_model(pretrained=True)
        elif self.config.model_type == "ensemble":
            ensemble, _ = create_ecg_ensemble(device=str(self.device))
            model = ensemble
        else:
            raise ValueError(f"Unknown model type: {self.config.model_type}")
        
        model = model.to(self.device)
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        self.logger.info(f"Model created with {total_params:,} total parameters")
        self.logger.info(f"Trainable parameters: {trainable_params:,}")
        
        return model
    
    def create_dataloaders(self) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """Create train, validation, and test dataloaders"""
        # Create datasets
        train_dataset = ECGDataset(self.data_dir, split="train", augment=True)
        val_dataset = ECGDataset(self.data_dir, split="val", augment=False)
        test_dataset = ECGDataset(self.data_dir, split="test", augment=False)
        
        # Create dataloaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
            drop_last=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )
        
        self.logger.info(f"Created dataloaders: Train={len(train_loader)}, Val={len(val_loader)}, Test={len(test_loader)}")
        
        return train_loader, val_loader, test_loader
    
    def create_optimizer(self, model: nn.Module) -> optim.Optimizer:
        """Create optimizer with model-specific parameter groups"""
        param_groups = []
        
        if self.config.model_type == "ensemble":
            # Different learning rates for different components
            ensemble_params = []
            individual_model_params = []
            
            for name, param in model.named_parameters():
                if 'models.' in name:
                    individual_model_params.append(param)
                else:
                    ensemble_params.append(param)
            
            param_groups = [
                {'params': individual_model_params, 'lr': self.config.learning_rate * 0.1},
                {'params': ensemble_params, 'lr': self.config.learning_rate}
            ]
        
        elif self.config.model_type in ["hubert", "vit_detr"]:
            # Different learning rates for backbone and head
            backbone_params = []
            head_params = []
            
            for name, param in model.named_parameters():
                if any(x in name.lower() for x in ['backbone', 'encoder', 'hubert', 'transformer']):
                    backbone_params.append(param)
                else:
                    head_params.append(param)
            
            param_groups = [
                {'params': backbone_params, 'lr': self.config.learning_rate * 0.1},
                {'params': head_params, 'lr': self.config.learning_rate}
            ]
        
        else:
            # Single learning rate for all parameters
            param_groups = [{'params': model.parameters(), 'lr': self.config.learning_rate}]
        
        # Create optimizer
        optimizer = optim.AdamW(
            param_groups,
            weight_decay=self.config.weight_decay,
            betas=(0.9, 0.999),
            eps=1e-8
        )
        
        return optimizer
    
    def create_scheduler(self, optimizer: optim.Optimizer, 
                        num_training_steps: int) -> optim.lr_scheduler._LRScheduler:
        """Create learning rate scheduler"""
        if self.config.scheduler_type == "cosine":
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer, 
                T_max=self.config.num_epochs,
                eta_min=self.config.learning_rate * 0.01
            )
        elif self.config.scheduler_type == "step":
            scheduler = optim.lr_scheduler.StepLR(
                optimizer,
                step_size=self.config.num_epochs // 3,
                gamma=0.1
            )
        elif self.config.scheduler_type == "warmup_cosine":
            def lr_lambda(current_step):
                if current_step < self.config.warmup_epochs:
                    return current_step / self.config.warmup_epochs
                else:
                    progress = (current_step - self.config.warmup_epochs) / (self.config.num_epochs - self.config.warmup_epochs)
                    return 0.5 * (1.0 + np.cos(np.pi * progress))
            
            scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        else:
            scheduler = optim.lr_scheduler.ConstantLR(optimizer, factor=1.0)
        
        return scheduler
    
    def compute_loss(self, model: nn.Module, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Compute loss for different model types"""
        images = batch['image'].to(self.device)
        masks = batch['mask'].to(self.device)
        
        # Prepare targets
        targets = {
            'keypoints': batch['pqrst_points'].to(self.device),
            'classes': batch['classes'].to(self.device),
            'confidence': batch['confidence'].to(self.device)
        }
        
        # Forward pass
        if self.config.model_type == "maskrcnn":
            # Mask R-CNN expects list of images
            image_list = [images[i] for i in range(images.size(0))]
            
            if model.training:
                # Prepare targets for Mask R-CNN training
                target_list = []
                for i in range(len(image_list)):
                    target = {
                        'boxes': torch.tensor([[0, 0, images.shape[3], images.shape[2]]], dtype=torch.float32).to(self.device),
                        'labels': torch.tensor([1], dtype=torch.int64).to(self.device),
                        'masks': masks[i].unsqueeze(0).byte(),
                        'keypoints': targets['keypoints'][i].unsqueeze(0)
                    }
                    # Add visibility to keypoints
                    keypoints_with_vis = torch.cat([
                        target['keypoints'], 
                        torch.ones(target['keypoints'].shape[:-1] + (1,)).to(self.device)
                    ], dim=-1)
                    target['keypoints'] = keypoints_with_vis
                    target_list.append(target)
                
                loss_dict = model(image_list, target_list)
                total_loss = sum(loss for loss in loss_dict.values())
                
                return {
                    'total_loss': total_loss,
                    **{k: v for k, v in loss_dict.items()}
                }
            else:
                predictions = model(image_list)
                # Simple loss for inference mode
                return {'total_loss': torch.tensor(0.0).to(self.device)}
        
        else:
            # Other model types
            if hasattr(model, 'compute_loss'):
                predictions = model(images, masks)
                losses = model.compute_loss(predictions, targets)
            elif self.config.model_type == "ensemble":
                predictions = model(images, masks)
                losses = model.compute_ensemble_loss(predictions, targets)
            else:
                # Generic loss computation
                predictions = model(images, masks)
                
                # Basic losses
                if 'keypoint_coords' in predictions:
                    keypoint_loss = F.mse_loss(predictions['keypoint_coords'], targets['keypoints'])
                else:
                    keypoint_loss = torch.tensor(0.0).to(self.device)
                
                if 'class_logits' in predictions:
                    class_loss = F.cross_entropy(predictions['class_logits'], targets['classes'])
                else:
                    class_loss = torch.tensor(0.0).to(self.device)
                
                if 'confidence_scores' in predictions:
                    confidence_loss = F.binary_cross_entropy(predictions['confidence_scores'], targets['confidence'])
                else:
                    confidence_loss = torch.tensor(0.0).to(self.device)
                
                total_loss = keypoint_loss + class_loss + confidence_loss
                
                losses = {
                    'total_loss': total_loss,
                    'keypoint_loss': keypoint_loss,
                    'class_loss': class_loss,
                    'confidence_loss': confidence_loss
                }
            
            return losses
    
    def train_epoch(self, model: nn.Module, dataloader: DataLoader, 
                   optimizer: optim.Optimizer, scheduler: optim.lr_scheduler._LRScheduler) -> Dict[str, float]:
        """Train for one epoch"""
        model.train()
        epoch_losses = defaultdict(float)
        num_batches = len(dataloader)
        
        progress_bar = tqdm(dataloader, desc=f"Epoch {self.current_epoch + 1}/{self.config.num_epochs}")
        
        for batch_idx, batch in enumerate(progress_bar):
            optimizer.zero_grad()
            
            try:
                if self.config.mixed_precision and self.scaler is not None:
                    with torch.cuda.amp.autocast():
                        losses = self.compute_loss(model, batch)
                    
                    self.scaler.scale(losses['total_loss']).backward()
                    
                    if self.config.gradient_clip_norm > 0:
                        self.scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), self.config.gradient_clip_norm)
                    
                    self.scaler.step(optimizer)
                    self.scaler.update()
                else:
                    losses = self.compute_loss(model, batch)
                    losses['total_loss'].backward()
                    
                    if self.config.gradient_clip_norm > 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), self.config.gradient_clip_norm)
                    
                    optimizer.step()
                
                # Accumulate losses
                for key, value in losses.items():
                    if isinstance(value, torch.Tensor):
                        epoch_losses[key] += value.item()
                
                # Update progress bar
                current_lr = optimizer.param_groups[0]['lr']
                progress_bar.set_postfix({
                    'loss': f"{losses['total_loss'].item():.4f}",
                    'lr': f"{current_lr:.2e}"
                })
                
                # Log to wandb
                if self.config.use_wandb and batch_idx % 10 == 0:
                    wandb.log({
                        'train/batch_loss': losses['total_loss'].item(),
                        'train/learning_rate': current_lr,
                        'train/epoch': self.current_epoch + batch_idx / num_batches
                    })
            
            except Exception as e:
                self.logger.error(f"Error in training batch {batch_idx}: {e}")
                continue
        
        # Update scheduler
        if scheduler is not None:
            scheduler.step()
        
        # Average losses
        avg_losses = {key: value / num_batches for key, value in epoch_losses.items()}
        
        return avg_losses
    
    def validate_epoch(self, model: nn.Module, dataloader: DataLoader) -> Dict[str, float]:
        """Validate for one epoch"""
        model.eval()
        epoch_losses = defaultdict(float)
        num_batches = len(dataloader)
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Validation"):
                try:
                    losses = self.compute_loss(model, batch)
                    
                    # Accumulate losses
                    for key, value in losses.items():
                        if isinstance(value, torch.Tensor):
                            epoch_losses[key] += value.item()
                
                except Exception as e:
                    self.logger.error(f"Error in validation batch: {e}")
                    continue
        
        # Average losses
        avg_losses = {key: value / num_batches for key, value in epoch_losses.items()}
        
        return avg_losses
    
    def save_checkpoint(self, model: nn.Module, optimizer: optim.Optimizer, 
                       scheduler: optim.lr_scheduler._LRScheduler, 
                       epoch: int, val_loss: float, is_best: bool = False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
            'val_loss': val_loss,
            'config': vars(self.config),
            'training_history': dict(self.training_history)
        }
        
        # Save regular checkpoint
        checkpoint_path = self.save_dir / f"checkpoint_epoch_{epoch}.pt"
        torch.save(checkpoint, checkpoint_path)
        
        # Save best model
        if is_best:
            best_path = self.save_dir / "best_model.pt"
            torch.save(checkpoint, best_path)
            self.logger.info(f"New best model saved with validation loss: {val_loss:.4f}")
        
        # Save latest model
        latest_path = self.save_dir / "latest_model.pt"
        torch.save(checkpoint, latest_path)
    
    def train(self) -> Dict[str, List[float]]:
        """Main training loop"""
        self.logger.info("Starting ECG PQRST detection training...")
        
        # Create model, dataloaders, optimizer, scheduler
        model = self.create_model()
        train_loader, val_loader, test_loader = self.create_dataloaders()
        optimizer = self.create_optimizer(model)
        scheduler = self.create_scheduler(optimizer, len(train_loader) * self.config.num_epochs)
        
        # Training loop
        for epoch in range(self.config.num_epochs):
            self.current_epoch = epoch
            
            # Training
            train_losses = self.train_epoch(model, train_loader, optimizer, scheduler)
            
            # Validation
            if epoch % self.config.validate_every_n_epochs == 0:
                val_losses = self.validate_epoch(model, val_loader)
                
                # Check for improvement
                val_loss = val_losses['total_loss']
                is_best = val_loss < self.best_val_loss
                
                if is_best:
                    self.best_val_loss = val_loss
                    self.patience_counter = 0
                else:
                    self.patience_counter += 1
                
                # Log metrics
                self.logger.info(f"Epoch {epoch + 1}/{self.config.num_epochs}")
                self.logger.info(f"Train Loss: {train_losses['total_loss']:.4f}")
                self.logger.info(f"Val Loss: {val_loss:.4f}")
                
                # Store history
                for key, value in train_losses.items():
                    self.training_history[f'train_{key}'].append(value)
                for key, value in val_losses.items():
                    self.training_history[f'val_{key}'].append(value)
                
                # Log to wandb
                if self.config.use_wandb:
                    log_dict = {}
                    for key, value in train_losses.items():
                        log_dict[f'train/{key}'] = value
                    for key, value in val_losses.items():
                        log_dict[f'val/{key}'] = value
                    log_dict['epoch'] = epoch
                    wandb.log(log_dict)
                
                # Save checkpoint
                if epoch % self.config.save_every_n_epochs == 0 or is_best:
                    self.save_checkpoint(model, optimizer, scheduler, epoch, val_loss, is_best)
                
                # Early stopping
                if self.patience_counter >= self.config.early_stopping_patience:
                    self.logger.info(f"Early stopping triggered after {epoch + 1} epochs")
                    break
        
        # Save final model
        self.save_checkpoint(model, optimizer, scheduler, self.current_epoch, val_losses['total_loss'])
        
        self.logger.info("Training completed!")
        
        return dict(self.training_history)

def create_training_configs() -> Dict[str, ECGTrainingConfig]:
    """Create training configurations for all model types"""
    configs = {}
    
    model_types = ["vit_detr", "multimodal", "hubert", "maskrcnn", "ensemble"]
    
    for model_type in model_types:
        config = ECGTrainingConfig(
            model_type=model_type,
            num_epochs=50,
            device="mps",
            use_wandb=False  # Set to True if you want to use wandb
        )
        configs[model_type] = config
    
    return configs

if __name__ == "__main__":
    # Example usage
    data_dir = Path("../data")
    save_dir = Path("../experiments")
    
    # Create configurations for all models
    configs = create_training_configs()
    
    # Train a specific model
    model_type = "ensemble"  # Change this to train different models
    
    if model_type in configs:
        config = configs[model_type]
        trainer = ECGAdvancedTrainer(config, save_dir / model_type, data_dir)
        
        print(f"Training {model_type} model...")
        history = trainer.train()
        
        print("Training completed successfully!")
        print(f"Best validation loss: {trainer.best_val_loss:.4f}")
    else:
        print(f"Unknown model type: {model_type}")
        print(f"Available types: {list(configs.keys())}")