#!/usr/bin/env python3
"""
Enhanced Data Augmentation for ECG Analysis
Implements advanced augmentation techniques for robust ECG model training.
"""

import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import numpy as np
import cv2
from PIL import Image, ImageFilter, ImageEnhance
import random
from typing import Tuple, Optional, Union
import albumentations as A
from albumentations.pytorch import ToTensorV2


class ECGAugmentationPipeline:
    """Advanced augmentation pipeline specifically designed for ECG images."""
    
    def __init__(self, 
                 image_size: Tuple[int, int] = (224, 224),
                 augmentation_strength: str = 'medium',
                 preserve_medical_integrity: bool = True):
        """
        Initialize ECG augmentation pipeline.
        
        Args:
            image_size: Target image size (height, width)
            augmentation_strength: 'light', 'medium', 'heavy'
            preserve_medical_integrity: If True, limits augmentations that could alter diagnostic features
        """
        self.image_size = image_size
        self.augmentation_strength = augmentation_strength
        self.preserve_medical_integrity = preserve_medical_integrity
        
        # ImageNet normalization (commonly used for pre-trained models)
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], 
            std=[0.229, 0.224, 0.225]
        )
        
        # Define augmentation parameters based on strength
        self.aug_params = self._get_augmentation_parameters()
        
    def _get_augmentation_parameters(self) -> dict:
        """Get augmentation parameters based on strength level."""
        if self.augmentation_strength == 'light':
            return {
                'rotation_degrees': 2,
                'translate': (0.01, 0.01),
                'scale': (0.98, 1.02),
                'brightness': 0.05,
                'contrast': 0.05,
                'saturation': 0.02,
                'hue': 0.01,
                'blur_prob': 0.1,
                'noise_prob': 0.1,
                'flip_prob': 0.1
            }
        elif self.augmentation_strength == 'medium':
            return {
                'rotation_degrees': 5,
                'translate': (0.02, 0.02),
                'scale': (0.95, 1.05),
                'brightness': 0.1,
                'contrast': 0.1,
                'saturation': 0.05,
                'hue': 0.02,
                'blur_prob': 0.2,
                'noise_prob': 0.2,
                'flip_prob': 0.3
            }
        else:  # heavy
            return {
                'rotation_degrees': 8,
                'translate': (0.03, 0.03),
                'scale': (0.9, 1.1),
                'brightness': 0.15,
                'contrast': 0.15,
                'saturation': 0.1,
                'hue': 0.03,
                'blur_prob': 0.3,
                'noise_prob': 0.3,
                'flip_prob': 0.4
            }
    
    def get_train_transforms(self) -> transforms.Compose:
        """Get training augmentation pipeline."""
        augmentations = [
            transforms.Resize(self.image_size),
            transforms.RandomApply([
                transforms.RandomAffine(
                    degrees=self.aug_params['rotation_degrees'],
                    translate=self.aug_params['translate'],
                    scale=self.aug_params['scale'],
                    fill=255  # White fill for ECG background
                )
            ], p=0.7),
            
            # Color augmentations (careful with medical images)
            transforms.RandomApply([
                transforms.ColorJitter(
                    brightness=self.aug_params['brightness'],
                    contrast=self.aug_params['contrast'],
                    saturation=self.aug_params['saturation'] if not self.preserve_medical_integrity else 0,
                    hue=self.aug_params['hue'] if not self.preserve_medical_integrity else 0
                )
            ], p=0.6),
            
            # Gaussian blur for noise robustness
            transforms.RandomApply([
                transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0))
            ], p=self.aug_params['blur_prob']),
            
            # Horizontal flip (only if medically appropriate)
            transforms.RandomHorizontalFlip(p=self.aug_params['flip_prob'] if not self.preserve_medical_integrity else 0),
            
            transforms.ToTensor(),
            
            # Add random noise
            RandomNoise(p=self.aug_params['noise_prob']),
            
            self.normalize
        ]
        
        return transforms.Compose(augmentations)
    
    def get_val_transforms(self) -> transforms.Compose:
        """Get validation/test transforms (minimal augmentation)."""
        return transforms.Compose([
            transforms.Resize(self.image_size),
            transforms.ToTensor(),
            self.normalize
        ])
    
    def get_albumentations_transforms(self) -> A.Compose:
        """Alternative augmentation using Albumentations library."""
        augmentations = [
            A.Resize(height=self.image_size[0], width=self.image_size[1]),
            
            A.OneOf([
                A.ShiftScaleRotate(
                    shift_limit=0.02,
                    scale_limit=0.05,
                    rotate_limit=self.aug_params['rotation_degrees'],
                    border_mode=cv2.BORDER_CONSTANT,
                    value=255,
                    p=0.7
                ),
                A.Perspective(scale=(0.02, 0.05), p=0.3)
            ], p=0.8),
            
            A.OneOf([
                A.GaussianBlur(blur_limit=(3, 5), p=0.3),
                A.MotionBlur(blur_limit=3, p=0.2),
                A.MedianBlur(blur_limit=3, p=0.1)
            ], p=self.aug_params['blur_prob']),
            
            A.RandomBrightnessContrast(
                brightness_limit=self.aug_params['brightness'],
                contrast_limit=self.aug_params['contrast'],
                p=0.6
            ),
            
            # Grid distortion for robustness (very light for medical images)
            A.GridDistortion(num_steps=2, distort_limit=0.1, p=0.1),
            
            # Elastic transform (very conservative)
            A.ElasticTransform(alpha=0.5, sigma=10, alpha_affine=5, p=0.05),
            
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ]
        
        if not self.preserve_medical_integrity:
            augmentations.insert(-2, A.HorizontalFlip(p=self.aug_params['flip_prob']))
        
        return A.Compose(augmentations)


class RandomNoise(nn.Module):
    """Add random Gaussian noise to tensor images."""
    
    def __init__(self, mean: float = 0.0, std: float = 0.01, p: float = 0.2):
        super().__init__()
        self.mean = mean
        self.std = std
        self.p = p
    
    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        if torch.rand(1) < self.p:
            noise = torch.randn_like(tensor) * self.std + self.mean
            return torch.clamp(tensor + noise, 0.0, 1.0)
        return tensor


class ECGSpecificAugmentations:
    """ECG-specific augmentations that preserve medical validity."""
    
    @staticmethod
    def add_grid_noise(image: np.ndarray, intensity: float = 0.1) -> np.ndarray:
        """Add realistic ECG grid noise."""
        h, w = image.shape[:2]
        
        # Create grid pattern
        grid = np.zeros_like(image)
        
        # Vertical lines
        for x in range(0, w, 20):  # Every 20 pixels
            if random.random() < intensity:
                grid[:, x:x+1] = random.randint(200, 230)
        
        # Horizontal lines  
        for y in range(0, h, 20):
            if random.random() < intensity:
                grid[y:y+1, :] = random.randint(200, 230)
        
        return np.clip(image.astype(float) + grid.astype(float) * 0.1, 0, 255).astype(np.uint8)
    
    @staticmethod
    def simulate_paper_artifacts(image: np.ndarray) -> np.ndarray:
        """Simulate paper/printing artifacts in ECG images."""
        # Add slight texture
        noise = np.random.normal(0, 2, image.shape)
        
        # Add occasional spots/marks
        if random.random() < 0.3:
            num_spots = random.randint(1, 5)
            for _ in range(num_spots):
                x = random.randint(0, image.shape[1]-5)
                y = random.randint(0, image.shape[0]-5)
                image[y:y+3, x:x+3] = random.randint(180, 220)
        
        return np.clip(image + noise, 0, 255).astype(np.uint8)
    
    @staticmethod
    def baseline_drift(signal: np.ndarray, max_drift: float = 0.1) -> np.ndarray:
        """Simulate baseline drift in ECG signals."""
        length = len(signal)
        drift = np.sin(np.linspace(0, 2*np.pi*random.uniform(0.5, 2), length)) * max_drift
        return signal + drift


class MixUpCutMix:
    """Implementation of MixUp and CutMix for ECG images."""
    
    def __init__(self, mixup_alpha: float = 0.2, cutmix_alpha: float = 1.0, prob: float = 0.5):
        self.mixup_alpha = mixup_alpha
        self.cutmix_alpha = cutmix_alpha
        self.prob = prob
    
    def mixup(self, x: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, float]:
        """Apply MixUp augmentation."""
        if self.mixup_alpha > 0:
            lam = np.random.beta(self.mixup_alpha, self.mixup_alpha)
        else:
            lam = 1
        
        batch_size = x.size(0)
        index = torch.randperm(batch_size)
        
        mixed_x = lam * x + (1 - lam) * x[index, :]
        y_a, y_b = y, y[index]
        
        return mixed_x, (y_a, y_b), lam
    
    def cutmix(self, x: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, float]:
        """Apply CutMix augmentation."""
        lam = np.random.beta(self.cutmix_alpha, self.cutmix_alpha)
        batch_size = x.size(0)
        index = torch.randperm(batch_size)
        
        # Generate random bounding box
        W, H = x.size(2), x.size(3)
        cut_rat = np.sqrt(1. - lam)
        cut_w = int(W * cut_rat)
        cut_h = int(H * cut_rat)
        
        cx = np.random.randint(W)
        cy = np.random.randint(H)
        
        bbx1 = np.clip(cx - cut_w // 2, 0, W)
        bby1 = np.clip(cy - cut_h // 2, 0, H)
        bbx2 = np.clip(cx + cut_w // 2, 0, W)
        bby2 = np.clip(cy + cut_h // 2, 0, H)
        
        x[:, :, bbx1:bbx2, bby1:bby2] = x[index, :, bbx1:bbx2, bby1:bby2]
        
        # Adjust lambda to exactly match pixel ratio
        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (W * H))
        
        y_a, y_b = y, y[index]
        return x, (y_a, y_b), lam
    
    def __call__(self, x: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, float]:
        """Apply MixUp or CutMix randomly."""
        if random.random() < self.prob:
            if random.random() < 0.5:
                return self.mixup(x, y)
            else:
                return self.cutmix(x, y)
        return x, y, 1.0


def create_ecg_augmentation_pipeline(
    image_size: Tuple[int, int] = (224, 224),
    augmentation_type: str = 'medium',
    use_albumentations: bool = False
) -> Union[transforms.Compose, A.Compose]:
    """
    Factory function to create ECG augmentation pipeline.
    
    Args:
        image_size: Target image dimensions
        augmentation_type: 'light', 'medium', 'heavy'
        use_albumentations: Whether to use Albumentations instead of torchvision
    
    Returns:
        Augmentation pipeline
    """
    pipeline = ECGAugmentationPipeline(
        image_size=image_size,
        augmentation_strength=augmentation_type,
        preserve_medical_integrity=True
    )
    
    if use_albumentations:
        return pipeline.get_albumentations_transforms()
    else:
        return pipeline.get_train_transforms()


# Example usage
if __name__ == "__main__":
    # Create augmentation pipeline
    train_transforms = create_ecg_augmentation_pipeline(
        image_size=(224, 224),
        augmentation_type='medium',
        use_albumentations=False
    )
    
    val_transforms = ECGAugmentationPipeline().get_val_transforms()
    
    print("✅ ECG Augmentation Pipeline Created")
    print(f"Training transforms: {len(train_transforms.transforms)} steps")
    print(f"Validation transforms: {len(val_transforms.transforms)} steps")