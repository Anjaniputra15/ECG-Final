#!/usr/bin/env python3
"""
Uncertainty Estimation for ECG Models
Implements Monte Carlo Dropout, Deep Ensembles, and Bayesian approaches
for reliable uncertainty quantification in medical AI.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Callable
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
from scipy import stats
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss


class MonteCarloDropout:
    """
    Monte Carlo Dropout for uncertainty estimation.
    Enables uncertainty quantification using dropout at inference time.
    """
    
    def __init__(self, model: nn.Module, num_samples: int = 100):
        """
        Initialize Monte Carlo Dropout.
        
        Args:
            model: The neural network model
            num_samples: Number of forward passes for uncertainty estimation
        """
        self.model = model
        self.num_samples = num_samples
        self.device = next(model.parameters()).device
        
    def enable_dropout(self):
        """Enable dropout layers during inference."""
        for module in self.model.modules():
            if isinstance(module, nn.Dropout):
                module.train()
    
    def disable_dropout(self):
        """Disable dropout layers (standard inference)."""
        for module in self.model.modules():
            if isinstance(module, nn.Dropout):
                module.eval()
    
    def predict_with_uncertainty(self, 
                                inputs: Union[torch.Tensor, Tuple, Dict],
                                return_individual_predictions: bool = False) -> Dict[str, torch.Tensor]:
        """
        Generate predictions with uncertainty estimates.
        
        Args:
            inputs: Model inputs (tensor, tuple, or dict)
            return_individual_predictions: Whether to return all individual predictions
            
        Returns:
            Dictionary containing predictions and uncertainty estimates
        """
        # Store original training state
        original_training_state = self.model.training
        
        # Enable dropout and set model to training mode
        self.model.train()
        self.enable_dropout()
        
        predictions = []
        confidence_scores = []
        
        with torch.no_grad():
            for _ in range(self.num_samples):
                if isinstance(inputs, dict):
                    output = self.model(**inputs)
                elif isinstance(inputs, (tuple, list)):
                    output = self.model(*inputs)
                else:
                    output = self.model(inputs)
                
                # Handle different output formats
                if isinstance(output, dict):
                    # Multi-output model
                    if 'diagnosis_logits' in output:
                        logits = output['diagnosis_logits']
                        confidences = output.get('confidence_scores', None)
                    elif 'class_logits' in output:
                        logits = output['class_logits']
                        confidences = output.get('confidence_scores', None)
                    else:
                        logits = list(output.values())[0]  # Take first output
                        confidences = None
                else:
                    logits = output
                    confidences = None
                
                # Apply softmax to get probabilities
                if logits.dim() > 2:
                    # Handle DETR-style outputs with multiple predictions
                    probs = torch.softmax(logits.flatten(0, 1), dim=-1)
                    probs = probs.view(logits.shape)
                else:
                    probs = torch.softmax(logits, dim=-1)
                
                predictions.append(probs)
                
                if confidences is not None:
                    confidence_scores.append(confidences)
        
        # Restore original training state
        self.model.train(original_training_state)
        
        # Stack predictions
        predictions = torch.stack(predictions, dim=0)  # [num_samples, batch_size, ...]
        
        # Calculate statistics
        mean_predictions = predictions.mean(0)
        
        # Predictive entropy (aleatoric + epistemic uncertainty)
        entropy = -torch.sum(mean_predictions * torch.log(mean_predictions + 1e-8), dim=-1)
        
        # Mutual information (epistemic uncertainty)
        individual_entropies = -torch.sum(predictions * torch.log(predictions + 1e-8), dim=-1)
        mean_entropy = individual_entropies.mean(0)
        mutual_info = entropy - mean_entropy
        
        # Variance-based uncertainty
        prediction_variance = predictions.var(0).sum(-1)  # Sum variance across classes
        
        # Confidence-based metrics
        max_prob_variance = predictions.max(-1)[0].var(0)  # Variance in max probability
        
        results = {
            'mean_predictions': mean_predictions,
            'prediction_variance': prediction_variance,
            'predictive_entropy': entropy,
            'mutual_information': mutual_info,
            'max_prob_variance': max_prob_variance,
            'prediction_std': predictions.std(0),
        }
        
        # Add confidence scores if available
        if confidence_scores:
            confidence_scores = torch.stack(confidence_scores, dim=0)
            mean_confidence = confidence_scores.mean(0)
            confidence_variance = confidence_scores.var(0)
            
            results.update({
                'mean_confidence': mean_confidence,
                'confidence_variance': confidence_variance
            })
        
        # Add individual predictions if requested
        if return_individual_predictions:
            results['individual_predictions'] = predictions
        
        return results


class DeepEnsemble:
    """
    Deep Ensemble for uncertainty estimation.
    Trains multiple models with different initializations.
    """
    
    def __init__(self, 
                 model_factory: Callable,
                 num_models: int = 5,
                 device: str = 'mps'):
        """
        Initialize Deep Ensemble.
        
        Args:
            model_factory: Function that creates a new model instance
            num_models: Number of models in ensemble
            device: Device to run models on
        """
        self.model_factory = model_factory
        self.num_models = num_models
        self.device = device
        self.models = []
        
        # Create ensemble models
        for i in range(num_models):
            model = model_factory()
            model.to(device)
            self.models.append(model)
    
    def train_ensemble(self, 
                      train_dataloader,
                      val_dataloader, 
                      num_epochs: int = 100,
                      loss_fn: nn.Module = None,
                      optimizer_factory: Callable = None):
        """
        Train all models in the ensemble.
        
        Args:
            train_dataloader: Training data loader
            val_dataloader: Validation data loader
            num_epochs: Number of training epochs
            loss_fn: Loss function
            optimizer_factory: Function to create optimizer for each model
        """
        if loss_fn is None:
            loss_fn = nn.CrossEntropyLoss()
        
        if optimizer_factory is None:
            optimizer_factory = lambda model: torch.optim.Adam(model.parameters(), lr=1e-4)
        
        ensemble_results = []
        
        for i, model in enumerate(self.models):
            print(f"Training ensemble model {i+1}/{self.num_models}")
            
            optimizer = optimizer_factory(model)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
            
            model_results = self._train_single_model(
                model, train_dataloader, val_dataloader, 
                num_epochs, loss_fn, optimizer, scheduler
            )
            
            ensemble_results.append(model_results)
        
        return ensemble_results
    
    def _train_single_model(self, model, train_loader, val_loader, 
                           num_epochs, loss_fn, optimizer, scheduler):
        """Train a single model in the ensemble."""
        train_losses = []
        val_losses = []
        
        for epoch in range(num_epochs):
            # Training
            model.train()
            epoch_train_loss = 0
            
            for batch in train_loader:
                optimizer.zero_grad()
                
                # Handle different input formats
                if isinstance(batch, dict):
                    inputs = {k: v.to(self.device) for k, v in batch.items() if k != 'targets'}
                    targets = batch['targets'].to(self.device)
                    outputs = model(**inputs)
                else:
                    inputs, targets = batch
                    inputs = inputs.to(self.device)
                    targets = targets.to(self.device)
                    outputs = model(inputs)
                
                # Handle different output formats
                if isinstance(outputs, dict):
                    logits = outputs.get('diagnosis_logits', outputs.get('class_logits', list(outputs.values())[0]))
                else:
                    logits = outputs
                
                loss = loss_fn(logits, targets)
                loss.backward()
                optimizer.step()
                
                epoch_train_loss += loss.item()
            
            # Validation
            model.eval()
            epoch_val_loss = 0
            
            with torch.no_grad():
                for batch in val_loader:
                    if isinstance(batch, dict):
                        inputs = {k: v.to(self.device) for k, v in batch.items() if k != 'targets'}
                        targets = batch['targets'].to(self.device)
                        outputs = model(**inputs)
                    else:
                        inputs, targets = batch
                        inputs = inputs.to(self.device)
                        targets = targets.to(self.device)
                        outputs = model(inputs)
                    
                    if isinstance(outputs, dict):
                        logits = outputs.get('diagnosis_logits', outputs.get('class_logits', list(outputs.values())[0]))
                    else:
                        logits = outputs
                    
                    loss = loss_fn(logits, targets)
                    epoch_val_loss += loss.item()
            
            scheduler.step()
            
            train_losses.append(epoch_train_loss / len(train_loader))
            val_losses.append(epoch_val_loss / len(val_loader))
            
            if (epoch + 1) % 10 == 0:
                print(f"  Epoch {epoch+1}: Train Loss = {train_losses[-1]:.4f}, Val Loss = {val_losses[-1]:.4f}")
        
        return {'train_losses': train_losses, 'val_losses': val_losses}
    
    def predict_with_uncertainty(self, inputs) -> Dict[str, torch.Tensor]:
        """Generate ensemble predictions with uncertainty."""
        predictions = []
        
        for model in self.models:
            model.eval()
            with torch.no_grad():
                if isinstance(inputs, dict):
                    output = model(**inputs)
                elif isinstance(inputs, (tuple, list)):
                    output = model(*inputs)
                else:
                    output = model(inputs)
                
                # Handle different output formats
                if isinstance(output, dict):
                    logits = output.get('diagnosis_logits', output.get('class_logits', list(output.values())[0]))
                else:
                    logits = output
                
                probs = torch.softmax(logits, dim=-1)
                predictions.append(probs)
        
        predictions = torch.stack(predictions, dim=0)
        
        # Calculate ensemble statistics
        mean_predictions = predictions.mean(0)
        prediction_variance = predictions.var(0).sum(-1)
        
        # Epistemic uncertainty (model uncertainty)
        epistemic_uncertainty = prediction_variance
        
        # Ensemble disagreement
        pairwise_kl_div = []
        for i in range(len(predictions)):
            for j in range(i+1, len(predictions)):
                kl_div = F.kl_div(
                    torch.log(predictions[i] + 1e-8), 
                    predictions[j], 
                    reduction='none'
                ).sum(-1)
                pairwise_kl_div.append(kl_div)
        
        ensemble_disagreement = torch.stack(pairwise_kl_div).mean(0)
        
        return {
            'mean_predictions': mean_predictions,
            'prediction_variance': prediction_variance,
            'epistemic_uncertainty': epistemic_uncertainty,
            'ensemble_disagreement': ensemble_disagreement,
            'individual_predictions': predictions
        }


class UncertaintyCalibrator:
    """
    Calibrates model uncertainty estimates to match true confidence.
    """
    
    def __init__(self):
        self.calibrator = None
        self.is_fitted = False
    
    def fit(self, 
            confidence_scores: np.ndarray,
            predictions: np.ndarray,
            true_labels: np.ndarray,
            method: str = 'platt'):
        """
        Fit calibration model.
        
        Args:
            confidence_scores: Model confidence scores
            predictions: Model predictions (probabilities)
            true_labels: True labels
            method: Calibration method ('platt', 'isotonic')
        """
        from sklearn.calibration import CalibratedClassifierCV
        from sklearn.linear_model import LogisticRegression
        from sklearn.isotonic import IsotonicRegression
        
        if method == 'platt':
            self.calibrator = LogisticRegression()
            self.calibrator.fit(confidence_scores.reshape(-1, 1), true_labels)
        elif method == 'isotonic':
            self.calibrator = IsotonicRegression(out_of_bounds='clip')
            self.calibrator.fit(confidence_scores, true_labels)
        
        self.is_fitted = True
    
    def calibrate(self, confidence_scores: np.ndarray) -> np.ndarray:
        """Apply calibration to confidence scores."""
        if not self.is_fitted:
            raise ValueError("Calibrator must be fitted before use")
        
        if hasattr(self.calibrator, 'predict_proba'):
            return self.calibrator.predict_proba(confidence_scores.reshape(-1, 1))[:, 1]
        else:
            return self.calibrator.predict(confidence_scores)
    
    def evaluate_calibration(self, 
                           confidence_scores: np.ndarray,
                           predictions: np.ndarray,
                           true_labels: np.ndarray,
                           n_bins: int = 10) -> Dict[str, float]:
        """
        Evaluate calibration quality.
        
        Returns:
            Dictionary with calibration metrics
        """
        # Reliability diagram data
        fraction_of_positives, mean_predicted_value = calibration_curve(
            true_labels, confidence_scores, n_bins=n_bins
        )
        
        # Expected Calibration Error (ECE)
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        ece = 0
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (confidence_scores > bin_lower) & (confidence_scores <= bin_upper)
            prop_in_bin = in_bin.mean()
            
            if prop_in_bin > 0:
                accuracy_in_bin = true_labels[in_bin].mean()
                avg_confidence_in_bin = confidence_scores[in_bin].mean()
                ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
        
        # Brier Score
        brier_score = brier_score_loss(true_labels, confidence_scores)
        
        return {
            'expected_calibration_error': ece,
            'brier_score': brier_score,
            'fraction_of_positives': fraction_of_positives,
            'mean_predicted_value': mean_predicted_value
        }


class UncertaintyVisualizer:
    """
    Visualization tools for uncertainty analysis.
    """
    
    @staticmethod
    def plot_uncertainty_distribution(uncertainty_dict: Dict[str, torch.Tensor],
                                    save_path: Optional[str] = None) -> plt.Figure:
        """Plot distribution of different uncertainty measures."""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle('Uncertainty Distribution Analysis', fontsize=16)
        
        # Convert tensors to numpy
        uncertainty_data = {}
        for key, value in uncertainty_dict.items():
            if isinstance(value, torch.Tensor):
                uncertainty_data[key] = value.detach().cpu().numpy().flatten()
        
        # Plot distributions
        metrics = ['prediction_variance', 'predictive_entropy', 'mutual_information', 'max_prob_variance']
        colors = ['blue', 'red', 'green', 'orange']
        
        for i, (metric, color) in enumerate(zip(metrics, colors)):
            if metric in uncertainty_data:
                ax = axes[i // 2, i % 2]
                data = uncertainty_data[metric]
                
                ax.hist(data, bins=50, alpha=0.7, color=color, density=True)
                ax.set_title(f'{metric.replace("_", " ").title()}')
                ax.set_xlabel('Uncertainty Value')
                ax.set_ylabel('Density')
                ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    @staticmethod
    def plot_reliability_diagram(confidence_scores: np.ndarray,
                               accuracies: np.ndarray,
                               n_bins: int = 10,
                               save_path: Optional[str] = None) -> plt.Figure:
        """Plot reliability diagram for calibration analysis."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Reliability diagram
        fraction_of_positives, mean_predicted_value = calibration_curve(
            accuracies, confidence_scores, n_bins=n_bins
        )
        
        ax1.plot([0, 1], [0, 1], 'k--', label='Perfect Calibration')
        ax1.plot(mean_predicted_value, fraction_of_positives, 'ro-', label='Model')
        ax1.set_xlabel('Mean Predicted Probability')
        ax1.set_ylabel('Fraction of Positives')
        ax1.set_title('Reliability Diagram')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Confidence histogram
        ax2.hist(confidence_scores, bins=n_bins, alpha=0.7, density=True, color='skyblue')
        ax2.set_xlabel('Confidence Score')
        ax2.set_ylabel('Density')
        ax2.set_title('Confidence Score Distribution')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    @staticmethod
    def plot_uncertainty_vs_accuracy(uncertainty_scores: np.ndarray,
                                   accuracies: np.ndarray,
                                   uncertainty_type: str = 'Total Uncertainty',
                                   save_path: Optional[str] = None) -> plt.Figure:
        """Plot uncertainty vs accuracy to validate uncertainty quality."""
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        
        # Scatter plot
        ax.scatter(uncertainty_scores, accuracies, alpha=0.6, s=20)
        
        # Add trend line
        z = np.polyfit(uncertainty_scores, accuracies, 1)
        p = np.poly1d(z)
        ax.plot(uncertainty_scores, p(uncertainty_scores), "r--", alpha=0.8)
        
        # Calculate correlation
        correlation = np.corrcoef(uncertainty_scores, accuracies)[0, 1]
        
        ax.set_xlabel(f'{uncertainty_type} Score')
        ax.set_ylabel('Accuracy')
        ax.set_title(f'Uncertainty vs Accuracy (r = {correlation:.3f})')
        ax.grid(True, alpha=0.3)
        
        # Add text box with correlation
        textstr = f'Correlation: {correlation:.3f}'
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=12,
                verticalalignment='top', bbox=props)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig


# Example usage and testing
if __name__ == "__main__":
    print("🎯 Uncertainty Estimation for ECG Models")
    print("=" * 50)
    
    # Test with dummy model and data
    class DummyECGModel(nn.Module):
        def __init__(self, num_classes=6):
            super().__init__()
            self.classifier = nn.Sequential(
                nn.Linear(100, 64),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(32, num_classes)
            )
        
        def forward(self, x):
            return {'diagnosis_logits': self.classifier(x)}
    
    # Create model and uncertainty estimator
    model = DummyECGModel()
    mc_dropout = MonteCarloDropout(model, num_samples=20)
    
    # Test uncertainty estimation
    dummy_input = torch.randn(8, 100)
    uncertainty_results = mc_dropout.predict_with_uncertainty(dummy_input)
    
    print("Monte Carlo Dropout Results:")
    for key, value in uncertainty_results.items():
        if isinstance(value, torch.Tensor):
            print(f"  {key}: {value.shape}")
    
    print(f"\nUncertainty Metrics Available:")
    print(f"  📊 Prediction Variance")
    print(f"  🎲 Predictive Entropy")
    print(f"  🧠 Mutual Information (Epistemic)")
    print(f"  📈 Max Probability Variance")
    print(f"  🔧 Calibration Tools")
    print(f"  📋 Visualization Suite")
    
    print("\n✅ Uncertainty estimation module ready!")
    
    # Test visualizer
    visualizer = UncertaintyVisualizer()
    
    # Create sample data for testing
    sample_uncertainty = {
        'prediction_variance': torch.randn(100) * 0.1 + 0.2,
        'predictive_entropy': torch.randn(100) * 0.05 + 0.15,
        'mutual_information': torch.randn(100) * 0.02 + 0.05,
        'max_prob_variance': torch.randn(100) * 0.03 + 0.08
    }
    
    fig = visualizer.plot_uncertainty_distribution(sample_uncertainty)
    print(f"📊 Uncertainty distribution plot created")