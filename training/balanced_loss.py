#!/usr/bin/env python3
"""
Balanced Loss Functions for ECG Classification
Handles class imbalance in ECG datasets with advanced loss functions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from sklearn.utils.class_weight import compute_class_weight
from collections import Counter
import math


class WeightedFocalLoss(nn.Module):
    """
    Focal Loss with class weights for handling severe class imbalance.
    Combines focal loss with weighted loss for optimal performance.
    """
    
    def __init__(self, 
                 alpha: Optional[torch.Tensor] = None,
                 gamma: float = 2.0,
                 reduction: str = 'mean',
                 ignore_index: int = -100):
        """
        Initialize Weighted Focal Loss.
        
        Args:
            alpha: Class weights tensor [num_classes]
            gamma: Focusing parameter (higher gamma = more focus on hard examples)
            reduction: Reduction method ('mean', 'sum', 'none')
            ignore_index: Index to ignore in loss computation
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.ignore_index = ignore_index
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute focal loss.
        
        Args:
            inputs: Model predictions [batch_size, num_classes]
            targets: Ground truth labels [batch_size]
            
        Returns:
            Computed loss
        """
        ce_loss = F.cross_entropy(inputs, targets, reduction='none', ignore_index=self.ignore_index)
        pt = torch.exp(-ce_loss)
        
        # Apply alpha weights if provided
        if self.alpha is not None:
            if self.alpha.device != inputs.device:
                self.alpha = self.alpha.to(inputs.device)
            
            alpha_t = self.alpha.gather(0, targets)
            focal_loss = alpha_t * (1 - pt) ** self.gamma * ce_loss
        else:
            focal_loss = (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class ClassBalancedLoss(nn.Module):
    """
    Class-Balanced Loss based on effective number of samples.
    Re-weights loss based on effective number of samples per class.
    """
    
    def __init__(self, 
                 samples_per_class: List[int],
                 beta: float = 0.9999,
                 gamma: float = 2.0,
                 loss_type: str = 'focal'):
        """
        Initialize Class-Balanced Loss.
        
        Args:
            samples_per_class: Number of samples for each class
            beta: Re-weighting hyperparameter (0 <= beta < 1)
            gamma: Focal loss gamma parameter
            loss_type: Type of base loss ('focal', 'ce', 'sigmoid')
        """
        super().__init__()
        
        # Calculate effective number of samples
        effective_num = 1.0 - np.power(beta, samples_per_class)
        weights = (1.0 - beta) / np.array(effective_num)
        weights = weights / np.sum(weights) * len(weights)
        
        self.weights = torch.tensor(weights, dtype=torch.float32)
        self.gamma = gamma
        self.loss_type = loss_type
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute class-balanced loss."""
        if self.weights.device != inputs.device:
            self.weights = self.weights.to(inputs.device)
        
        if self.loss_type == 'focal':
            return self._focal_loss(inputs, targets)
        elif self.loss_type == 'ce':
            return F.cross_entropy(inputs, targets, weight=self.weights)
        elif self.loss_type == 'sigmoid':
            return self._sigmoid_focal_loss(inputs, targets)
    
    def _focal_loss(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Focal loss with class balancing."""
        ce_loss = F.cross_entropy(inputs, targets, weight=self.weights, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()
    
    def _sigmoid_focal_loss(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Sigmoid focal loss for multi-label classification."""
        p = torch.sigmoid(inputs)
        ce_loss = F.binary_cross_entropy_with_logits(inputs, targets.float(), reduction='none')
        p_t = p * targets + (1 - p) * (1 - targets)
        loss = ce_loss * ((1 - p_t) ** self.gamma)
        
        # Apply class weights
        alpha_t = self.weights[targets.long()]
        loss = alpha_t * loss
        
        return loss.mean()


class LDAMLoss(nn.Module):
    """
    Label-Distribution-Aware Margin Loss for imbalanced classification.
    Adjusts margins based on label frequency.
    """
    
    def __init__(self, 
                 cls_num_list: List[int],
                 max_m: float = 0.5,
                 weight: Optional[torch.Tensor] = None,
                 s: float = 30):
        """
        Initialize LDAM Loss.
        
        Args:
            cls_num_list: Number of samples per class
            max_m: Maximum margin
            weight: Class weights
            s: Scale parameter
        """
        super().__init__()
        
        m_list = 1.0 / np.sqrt(np.sqrt(cls_num_list))
        m_list = m_list * (max_m / np.max(m_list))
        m_list = torch.tensor(m_list, dtype=torch.float32)
        
        self.m_list = m_list
        self.s = s
        self.weight = weight
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute LDAM loss."""
        if self.m_list.device != inputs.device:
            self.m_list = self.m_list.to(inputs.device)
        
        batch_m = self.m_list[targets]
        batch_m = batch_m.view((-1, 1))
        
        x_m = inputs - batch_m
        
        # Create one-hot encoding
        index = torch.zeros_like(inputs, dtype=torch.uint8)
        index.scatter_(1, targets.data.view(-1, 1), 1)
        
        output = torch.where(index, x_m, inputs)
        return F.cross_entropy(self.s * output, targets, weight=self.weight)


class ECGLossCalculator:
    """
    Comprehensive loss calculator for ECG datasets with class imbalance handling.
    """
    
    def __init__(self, dataset_stats: Dict[str, Union[List, int, float]]):
        """
        Initialize ECG loss calculator.
        
        Args:
            dataset_stats: Dictionary containing dataset statistics
                - 'class_counts': List of sample counts per class
                - 'total_samples': Total number of samples
                - 'num_classes': Number of classes
        """
        self.class_counts = dataset_stats['class_counts']
        self.total_samples = dataset_stats['total_samples']
        self.num_classes = dataset_stats['num_classes']
        
        # Calculate various weighting schemes
        self.class_weights = self._calculate_class_weights()
        self.inverse_weights = self._calculate_inverse_weights()
        self.sqrt_inverse_weights = self._calculate_sqrt_inverse_weights()
        
    def _calculate_class_weights(self) -> torch.Tensor:
        """Calculate standard class weights (inverse frequency)."""
        weights = []
        for count in self.class_counts:
            if count > 0:
                weight = self.total_samples / (self.num_classes * count)
            else:
                weight = 1.0
            weights.append(weight)
        
        weights = np.array(weights)
        weights = weights / weights.sum() * len(weights)  # Normalize
        return torch.tensor(weights, dtype=torch.float32)
    
    def _calculate_inverse_weights(self) -> torch.Tensor:
        """Calculate simple inverse frequency weights."""
        weights = [1.0 / max(count, 1) for count in self.class_counts]
        weights = np.array(weights)
        weights = weights / weights.sum() * len(weights)
        return torch.tensor(weights, dtype=torch.float32)
    
    def _calculate_sqrt_inverse_weights(self) -> torch.Tensor:
        """Calculate square root inverse frequency weights (less aggressive)."""
        weights = [1.0 / np.sqrt(max(count, 1)) for count in self.class_counts]
        weights = np.array(weights)
        weights = weights / weights.sum() * len(weights)
        return torch.tensor(weights, dtype=torch.float32)
    
    def get_weighted_focal_loss(self, gamma: float = 2.0) -> WeightedFocalLoss:
        """Get weighted focal loss with calculated class weights."""
        return WeightedFocalLoss(alpha=self.class_weights, gamma=gamma)
    
    def get_class_balanced_loss(self, beta: float = 0.9999, gamma: float = 2.0) -> ClassBalancedLoss:
        """Get class-balanced loss."""
        return ClassBalancedLoss(self.class_counts, beta=beta, gamma=gamma)
    
    def get_ldam_loss(self, max_m: float = 0.5, s: float = 30) -> LDAMLoss:
        """Get LDAM loss."""
        return LDAMLoss(self.class_counts, max_m=max_m, weight=self.class_weights, s=s)
    
    def get_weighted_bce_loss(self) -> nn.BCEWithLogitsLoss:
        """Get weighted binary cross-entropy loss for multi-label classification."""
        # For multi-label ECG classification
        pos_weights = self.class_weights.clone()
        return nn.BCEWithLogitsLoss(pos_weight=pos_weights)


class AdaptiveLossScheduler:
    """
    Adaptive loss function scheduler that changes loss functions during training.
    """
    
    def __init__(self, 
                 loss_calculator: ECGLossCalculator,
                 warmup_epochs: int = 10,
                 transition_epochs: int = 50):
        """
        Initialize adaptive loss scheduler.
        
        Args:
            loss_calculator: ECG loss calculator
            warmup_epochs: Epochs to use standard weighted CE
            transition_epochs: Epochs to transition to focal loss
        """
        self.loss_calculator = loss_calculator
        self.warmup_epochs = warmup_epochs
        self.transition_epochs = transition_epochs
        
        # Initialize loss functions
        self.weighted_ce = nn.CrossEntropyLoss(weight=loss_calculator.class_weights)
        self.focal_loss = loss_calculator.get_weighted_focal_loss(gamma=1.0)
        self.strong_focal_loss = loss_calculator.get_weighted_focal_loss(gamma=2.0)
        
        self.current_epoch = 0
    
    def step(self, epoch: int):
        """Update current epoch."""
        self.current_epoch = epoch
    
    def get_current_loss(self) -> nn.Module:
        """Get loss function for current epoch."""
        if self.current_epoch < self.warmup_epochs:
            return self.weighted_ce
        elif self.current_epoch < self.transition_epochs:
            return self.focal_loss
        else:
            return self.strong_focal_loss
    
    def compute_loss(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute loss using current loss function."""
        loss_fn = self.get_current_loss()
        return loss_fn(predictions, targets)


def calculate_dataset_statistics(labels: List[int], num_classes: int) -> Dict[str, Union[List, int, float]]:
    """
    Calculate dataset statistics for loss function initialization.
    
    Args:
        labels: List of all labels in dataset
        num_classes: Total number of classes
        
    Returns:
        Dictionary with dataset statistics
    """
    label_counter = Counter(labels)
    class_counts = [label_counter.get(i, 0) for i in range(num_classes)]
    total_samples = len(labels)
    
    # Calculate class imbalance ratio
    max_count = max(class_counts)
    min_count = min([c for c in class_counts if c > 0])
    imbalance_ratio = max_count / min_count if min_count > 0 else float('inf')
    
    return {
        'class_counts': class_counts,
        'total_samples': total_samples,
        'num_classes': num_classes,
        'imbalance_ratio': imbalance_ratio,
        'class_distribution': {i: count/total_samples for i, count in enumerate(class_counts)}
    }


def create_balanced_loss_function(dataset_stats: Dict, 
                                loss_type: str = 'focal',
                                **kwargs) -> nn.Module:
    """
    Factory function to create balanced loss functions.
    
    Args:
        dataset_stats: Dataset statistics
        loss_type: Type of loss function
        **kwargs: Additional arguments for loss function
        
    Returns:
        Configured loss function
    """
    calculator = ECGLossCalculator(dataset_stats)
    
    if loss_type == 'focal':
        return calculator.get_weighted_focal_loss(**kwargs)
    elif loss_type == 'class_balanced':
        return calculator.get_class_balanced_loss(**kwargs)
    elif loss_type == 'ldam':
        return calculator.get_ldam_loss(**kwargs)
    elif loss_type == 'weighted_ce':
        return nn.CrossEntropyLoss(weight=calculator.class_weights)
    elif loss_type == 'weighted_bce':
        return calculator.get_weighted_bce_loss()
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")


# Example usage and testing
if __name__ == "__main__":
    # Example dataset statistics (ECG arrhythmia classification)
    example_stats = {
        'class_counts': [8000, 2000, 500, 100, 50, 10],  # Highly imbalanced
        'total_samples': 10660,
        'num_classes': 6
    }
    
    print("🏥 ECG Balanced Loss Functions")
    print("=" * 50)
    
    # Create loss calculator
    calculator = ECGLossCalculator(example_stats)
    
    print(f"Dataset Statistics:")
    print(f"  Total samples: {example_stats['total_samples']:,}")
    print(f"  Class distribution: {[f'{c:,}' for c in example_stats['class_counts']]}")
    print(f"  Imbalance ratio: {max(example_stats['class_counts'])/min([c for c in example_stats['class_counts'] if c > 0]):.1f}:1")
    
    # Test different loss functions
    print(f"\nAvailable Loss Functions:")
    print(f"  ✅ Weighted Focal Loss")
    print(f"  ✅ Class-Balanced Loss") 
    print(f"  ✅ LDAM Loss")
    print(f"  ✅ Weighted BCE Loss")
    print(f"  ✅ Adaptive Loss Scheduler")
    
    # Create sample predictions and targets
    batch_size = 32
    num_classes = 6
    predictions = torch.randn(batch_size, num_classes)
    targets = torch.randint(0, num_classes, (batch_size,))
    
    # Test focal loss
    focal_loss = calculator.get_weighted_focal_loss(gamma=2.0)
    loss_value = focal_loss(predictions, targets)
    print(f"\nWeighted Focal Loss: {loss_value.item():.4f}")
    
    print("\n✅ Balanced loss functions ready for training!")