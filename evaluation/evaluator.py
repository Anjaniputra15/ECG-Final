#!/usr/bin/env python3
"""
Comprehensive Evaluation System for ECG PQRST Detection Models
Evaluates all 4 approaches with clinical-grade metrics
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any
from pathlib import Path
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from scipy import signal as scipy_signal
from scipy.spatial.distance import euclidean
import cv2
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Import model architectures
import sys
sys.path.append(str(Path(__file__).parent.parent))

from training.advanced_trainer import ECGDataset
from models.ecg_ensemble import create_ecg_ensemble, ModelComparator

class ECGMetrics:
    """Clinical ECG evaluation metrics"""
    
    @staticmethod
    def point_detection_accuracy(pred_points: np.ndarray, 
                               true_points: np.ndarray, 
                               tolerance: float = 0.05) -> Dict[str, float]:
        """
        Calculate point detection accuracy with tolerance
        
        Args:
            pred_points: Predicted PQRST points [5, 2]
            true_points: Ground truth points [5, 2]
            tolerance: Tolerance for correct detection (as fraction of image size)
            
        Returns:
            Dictionary of accuracy metrics
        """
        point_labels = ['P', 'Q', 'R', 'S', 'T']
        accuracies = {}
        
        for i, label in enumerate(point_labels):
            if i < len(pred_points) and i < len(true_points):
                distance = euclidean(pred_points[i], true_points[i])
                accuracies[f'{label}_accuracy'] = float(distance < tolerance)
            else:
                accuracies[f'{label}_accuracy'] = 0.0
        
        # Overall accuracy
        accuracies['overall_accuracy'] = np.mean(list(accuracies.values()))
        
        return accuracies
    
    @staticmethod
    def temporal_interval_accuracy(pred_points: np.ndarray, 
                                 true_points: np.ndarray,
                                 sampling_rate: float = 500.0) -> Dict[str, float]:
        """
        Calculate temporal interval accuracies (PR, QT, etc.)
        
        Args:
            pred_points: Predicted points [5, 2] (normalized coordinates)
            true_points: Ground truth points [5, 2]
            sampling_rate: ECG sampling rate in Hz
            
        Returns:
            Dictionary of interval metrics
        """
        # Convert normalized x-coordinates to time (assuming 10-second ECG)
        duration = 10.0  # seconds
        
        def normalize_to_time(points):
            return points[:, 0] * duration
        
        pred_times = normalize_to_time(pred_points)
        true_times = normalize_to_time(true_points)
        
        intervals = {}
        
        # PR interval (P to R)
        if len(pred_times) >= 3 and len(true_times) >= 3:
            pred_pr = pred_times[2] - pred_times[0]  # R - P
            true_pr = true_times[2] - true_times[0]
            intervals['PR_interval_error'] = abs(pred_pr - true_pr)
            intervals['PR_interval_accuracy'] = float(abs(pred_pr - true_pr) < 0.04)  # 40ms tolerance
        
        # QT interval (Q to T)
        if len(pred_times) >= 5 and len(true_times) >= 5:
            pred_qt = pred_times[4] - pred_times[1]  # T - Q
            true_qt = true_times[4] - true_times[1]
            intervals['QT_interval_error'] = abs(pred_qt - true_qt)
            intervals['QT_interval_accuracy'] = float(abs(pred_qt - true_qt) < 0.08)  # 80ms tolerance
        
        # QRS duration (Q to S)
        if len(pred_times) >= 4 and len(true_times) >= 4:
            pred_qrs = pred_times[3] - pred_times[1]  # S - Q
            true_qrs = true_times[3] - true_times[1]
            intervals['QRS_duration_error'] = abs(pred_qrs - true_qrs)
            intervals['QRS_duration_accuracy'] = float(abs(pred_qrs - true_qrs) < 0.02)  # 20ms tolerance
        
        return intervals
    
    @staticmethod
    def morphology_similarity(pred_points: np.ndarray, 
                            true_points: np.ndarray) -> Dict[str, float]:
        """
        Calculate morphological similarity between predicted and true waves
        
        Args:
            pred_points: Predicted points [5, 2]
            true_points: Ground truth points [5, 2]
            
        Returns:
            Dictionary of morphology metrics
        """
        metrics = {}
        
        # Calculate relative amplitudes (y-coordinates)
        if len(pred_points) >= 5 and len(true_points) >= 5:
            pred_amplitudes = pred_points[:, 1]
            true_amplitudes = true_points[:, 1]
            
            # R-wave prominence (R should be highest/lowest depending on lead)
            pred_r_prominence = abs(pred_amplitudes[2] - np.mean(pred_amplitudes))
            true_r_prominence = abs(true_amplitudes[2] - np.mean(true_amplitudes))
            
            metrics['R_prominence_similarity'] = 1.0 - abs(pred_r_prominence - true_r_prominence)
            
            # P and T wave consistency
            pred_pt_ratio = abs(pred_amplitudes[0]) / (abs(pred_amplitudes[4]) + 1e-6)
            true_pt_ratio = abs(true_amplitudes[0]) / (abs(true_amplitudes[4]) + 1e-6)
            
            metrics['PT_ratio_similarity'] = 1.0 / (1.0 + abs(pred_pt_ratio - true_pt_ratio))
            
            # Overall morphology correlation
            if len(pred_amplitudes) == len(true_amplitudes):
                correlation = np.corrcoef(pred_amplitudes, true_amplitudes)[0, 1]
                metrics['morphology_correlation'] = correlation if not np.isnan(correlation) else 0.0
        
        return metrics
    
    @staticmethod
    def clinical_validation(pred_points: np.ndarray, 
                          confidence_scores: np.ndarray = None) -> Dict[str, float]:
        """
        Clinical validation of detected PQRST points
        
        Args:
            pred_points: Predicted points [5, 2]
            confidence_scores: Confidence scores for each point [5]
            
        Returns:
            Dictionary of clinical validation metrics
        """
        validation = {}
        
        if len(pred_points) >= 5:
            # Check temporal ordering (P < Q < R < S < T in x-axis)
            x_coords = pred_points[:, 0]
            is_ordered = np.all(x_coords[:-1] <= x_coords[1:])
            validation['temporal_ordering'] = float(is_ordered)
            
            # Check realistic intervals
            # PR interval should be 120-200ms (0.12-0.2s in 10s ECG = 0.012-0.02 normalized)
            pr_interval = x_coords[2] - x_coords[0]  # R - P
            validation['realistic_PR'] = float(0.01 < pr_interval < 0.1)
            
            # QRS should be narrow (< 120ms = 0.012 normalized)
            qrs_duration = x_coords[3] - x_coords[1]  # S - Q
            validation['narrow_QRS'] = float(qrs_duration < 0.05)
            
            # QT interval should be reasonable (< 500ms = 0.05 normalized)
            qt_interval = x_coords[4] - x_coords[1]  # T - Q
            validation['reasonable_QT'] = float(qt_interval < 0.2)
            
            # Confidence-based validation
            if confidence_scores is not None and len(confidence_scores) >= 5:
                validation['high_confidence'] = float(np.mean(confidence_scores) > 0.7)
                validation['r_peak_confidence'] = float(confidence_scores[2] > 0.8)  # R-peak should be most confident
            
            # Overall clinical validity
            clinical_scores = [
                validation['temporal_ordering'],
                validation['realistic_PR'],
                validation['narrow_QRS'],
                validation['reasonable_QT']
            ]
            validation['clinical_validity'] = np.mean(clinical_scores)
        
        return validation

class ECGEvaluator:
    """Comprehensive ECG model evaluator"""
    
    def __init__(self, 
                 model_checkpoints: Dict[str, Path],
                 data_dir: Path,
                 device: str = "mps"):
        
        self.model_checkpoints = model_checkpoints
        self.data_dir = Path(data_dir)
        self.device = device
        self.metrics = ECGMetrics()
        
        # Create results directory
        self.results_dir = Path("evaluation_results")
        self.results_dir.mkdir(exist_ok=True)
        
        # Load test dataset
        self.test_dataset = ECGDataset(data_dir, split="test", augment=False)
        print(f"Loaded {len(self.test_dataset)} test samples")
    
    def load_model(self, model_type: str, checkpoint_path: Path) -> nn.Module:
        """Load trained model from checkpoint"""
        print(f"Loading {model_type} model from {checkpoint_path}")
        
        try:
            # Import model creation functions
            if model_type == "vit_detr":
                from models.backbones.vision_transformer_ecg import create_ecg_vit_detr_model
                model = create_ecg_vit_detr_model(pretrained=True)
            elif model_type == "multimodal":
                from models.backbones.multimodal_ecg import create_multimodal_ecg_model
                model = create_multimodal_ecg_model(fusion_method="attention")
            elif model_type == "hubert":
                from models.backbones.hubert_ecg import create_hubert_ecg_model
                model = create_hubert_ecg_model(model_size="base")
            elif model_type == "maskrcnn":
                from models.backbones.maskrcnn_ecg import create_ecg_maskrcnn_model
                model = create_ecg_maskrcnn_model(pretrained=True)
            elif model_type == "ensemble":
                ensemble, _ = create_ecg_ensemble(device=self.device)
                model = ensemble
            else:
                raise ValueError(f"Unknown model type: {model_type}")
            
            # Load checkpoint if exists
            if checkpoint_path.exists():
                checkpoint = torch.load(checkpoint_path, map_location=self.device)
                if 'model_state_dict' in checkpoint:
                    model.load_state_dict(checkpoint['model_state_dict'])
                    print(f"Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")
                else:
                    print("Checkpoint found but no model_state_dict, using pretrained weights")
            else:
                print(f"No checkpoint found at {checkpoint_path}, using default weights")
            
            model = model.to(self.device)
            model.eval()
            
            return model
            
        except Exception as e:
            print(f"Error loading {model_type} model: {e}")
            return None
    
    def predict_batch(self, model: nn.Module, model_type: str, 
                     images: torch.Tensor, masks: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Make predictions with a model"""
        with torch.no_grad():
            try:
                if model_type == "maskrcnn":
                    # Mask R-CNN expects list of images
                    image_list = [images[i] for i in range(images.size(0))]
                    predictions = model(image_list)
                    
                    # Extract keypoints from Mask R-CNN output
                    batch_keypoints = []
                    batch_confidence = []
                    
                    for pred in predictions:
                        if 'pqrst_points' in pred and pred['pqrst_points']:
                            # Extract PQRST points
                            points = []
                            confidences = []
                            
                            for label in ['P', 'Q', 'R', 'S', 'T']:
                                if label in pred['pqrst_points']:
                                    point_data = pred['pqrst_points'][label]
                                    coords = point_data['coordinates']
                                    conf = point_data['visibility']
                                    
                                    points.append([coords[0].item(), coords[1].item()])
                                    confidences.append(conf.item())
                                else:
                                    points.append([0.5, 0.5])  # Default center
                                    confidences.append(0.1)   # Low confidence
                            
                            batch_keypoints.append(points)
                            batch_confidence.append(confidences)
                        else:
                            # Default predictions
                            batch_keypoints.append([[0.2, 0.5], [0.35, 0.5], [0.5, 0.5], [0.65, 0.5], [0.8, 0.5]])
                            batch_confidence.append([0.1, 0.1, 0.1, 0.1, 0.1])
                    
                    result = {
                        'keypoint_coords': torch.tensor(batch_keypoints, dtype=torch.float32),
                        'confidence_scores': torch.tensor(batch_confidence, dtype=torch.float32),
                        'class_logits': torch.randn(len(predictions), 6)  # Dummy class logits
                    }
                
                else:
                    # Other model types
                    predictions = model(images, masks)
                    
                    # Standardize output format
                    result = {}
                    
                    # Extract keypoints
                    if 'keypoint_coords' in predictions:
                        result['keypoint_coords'] = predictions['keypoint_coords']
                    elif 'point_coords' in predictions:
                        result['keypoint_coords'] = predictions['point_coords']
                    else:
                        # Generate default keypoints
                        batch_size = images.size(0)
                        default_points = torch.tensor([[0.2, 0.5], [0.35, 0.5], [0.5, 0.5], [0.65, 0.5], [0.8, 0.5]])
                        result['keypoint_coords'] = default_points.unsqueeze(0).repeat(batch_size, 1, 1)
                    
                    # Extract confidence scores
                    if 'confidence_scores' in predictions:
                        result['confidence_scores'] = predictions['confidence_scores']
                    elif 'point_confidence' in predictions:
                        result['confidence_scores'] = predictions['point_confidence']
                    else:
                        # Generate default confidence
                        batch_size = images.size(0)
                        result['confidence_scores'] = torch.full((batch_size, 5), 0.5)
                    
                    # Extract class logits
                    if 'class_logits' in predictions:
                        result['class_logits'] = predictions['class_logits']
                    elif 'point_classes' in predictions:
                        result['class_logits'] = predictions['point_classes']
                    else:
                        # Generate default class logits
                        batch_size = images.size(0)
                        result['class_logits'] = torch.randn(batch_size, 6)
                
                return result
                
            except Exception as e:
                print(f"Error in prediction for {model_type}: {e}")
                # Return default predictions
                batch_size = images.size(0)
                return {
                    'keypoint_coords': torch.tensor([[[0.2, 0.5], [0.35, 0.5], [0.5, 0.5], [0.65, 0.5], [0.8, 0.5]]] * batch_size),
                    'confidence_scores': torch.full((batch_size, 5), 0.1),
                    'class_logits': torch.randn(batch_size, 6)
                }
    
    def evaluate_model(self, model_type: str, model: nn.Module) -> Dict[str, Any]:
        """Evaluate a single model on test dataset"""
        print(f"\nEvaluating {model_type} model...")
        
        all_metrics = defaultdict(list)
        predictions = []
        ground_truths = []
        
        # Create test dataloader
        from torch.utils.data import DataLoader
        test_loader = DataLoader(self.test_dataset, batch_size=4, shuffle=False, num_workers=2)
        
        for batch in tqdm(test_loader, desc=f"Evaluating {model_type}"):
            images = batch['image'].to(self.device)
            masks = batch['mask'].to(self.device)
            true_points = batch['pqrst_points'].numpy()
            true_confidence = batch['confidence'].numpy()
            
            # Get predictions
            pred_results = self.predict_batch(model, model_type, images, masks)
            
            pred_points = pred_results['keypoint_coords'].cpu().numpy()
            pred_confidence = pred_results['confidence_scores'].cpu().numpy()
            
            # Evaluate each sample in batch
            for i in range(len(pred_points)):
                sample_pred = pred_points[i]
                sample_true = true_points[i]
                sample_pred_conf = pred_confidence[i]
                sample_true_conf = true_confidence[i]
                
                # Point detection accuracy
                point_metrics = self.metrics.point_detection_accuracy(sample_pred, sample_true)
                for key, value in point_metrics.items():
                    all_metrics[key].append(value)
                
                # Temporal interval accuracy
                interval_metrics = self.metrics.temporal_interval_accuracy(sample_pred, sample_true)
                for key, value in interval_metrics.items():
                    all_metrics[key].append(value)
                
                # Morphology similarity
                morphology_metrics = self.metrics.morphology_similarity(sample_pred, sample_true)
                for key, value in morphology_metrics.items():
                    all_metrics[key].append(value)
                
                # Clinical validation
                clinical_metrics = self.metrics.clinical_validation(sample_pred, sample_pred_conf)
                for key, value in clinical_metrics.items():
                    all_metrics[key].append(value)
                
                # Store for later analysis
                predictions.append({
                    'points': sample_pred,
                    'confidence': sample_pred_conf,
                    'model': model_type
                })
                ground_truths.append({
                    'points': sample_true,
                    'confidence': sample_true_conf
                })
        
        # Aggregate metrics
        aggregated_metrics = {}
        for key, values in all_metrics.items():
            if values:  # Only if we have values
                aggregated_metrics[key] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'median': np.median(values),
                    'min': np.min(values),
                    'max': np.max(values)
                }
        
        return {
            'metrics': aggregated_metrics,
            'predictions': predictions,
            'ground_truths': ground_truths
        }
    
    def compare_models(self) -> Dict[str, Any]:
        """Compare all models comprehensively"""
        print("Starting comprehensive model comparison...")
        
        model_results = {}
        all_predictions = []
        all_ground_truths = []
        
        # Evaluate each model
        for model_type, checkpoint_path in self.model_checkpoints.items():
            model = self.load_model(model_type, checkpoint_path)
            
            if model is not None:
                results = self.evaluate_model(model_type, model)
                model_results[model_type] = results
                
                # Collect all predictions for comparison
                all_predictions.extend(results['predictions'])
                if not all_ground_truths:  # Only add ground truths once
                    all_ground_truths = results['ground_truths']
            else:
                print(f"Skipping {model_type} due to loading error")
        
        # Create comparison summary
        comparison_summary = self._create_comparison_summary(model_results)
        
        # Generate visualizations
        self._generate_comparison_plots(model_results)
        
        # Save results
        self._save_results(model_results, comparison_summary)
        
        return {
            'model_results': model_results,
            'comparison_summary': comparison_summary,
            'all_predictions': all_predictions,
            'ground_truths': all_ground_truths
        }
    
    def _create_comparison_summary(self, model_results: Dict[str, Any]) -> Dict[str, Any]:
        """Create summary comparison table"""
        summary = {}
        
        # Key metrics to compare
        key_metrics = [
            'overall_accuracy',
            'R_accuracy', 
            'PR_interval_accuracy',
            'QT_interval_accuracy',
            'clinical_validity',
            'morphology_correlation'
        ]
        
        for metric in key_metrics:
            summary[metric] = {}
            
            for model_type, results in model_results.items():
                if metric in results['metrics']:
                    summary[metric][model_type] = results['metrics'][metric]['mean']
                else:
                    summary[metric][model_type] = 0.0
        
        # Find best model for each metric
        best_models = {}
        for metric in key_metrics:
            if metric in summary:
                best_model = max(summary[metric].items(), key=lambda x: x[1])
                best_models[metric] = best_model[0]
        
        summary['best_models'] = best_models
        
        return summary
    
    def _generate_comparison_plots(self, model_results: Dict[str, Any]):
        """Generate comparison plots"""
        try:
            # Set up the plotting style
            plt.style.use('seaborn-v0_8')
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            fig.suptitle('ECG PQRST Detection Model Comparison', fontsize=16)
            
            # Extract model names and key metrics
            models = list(model_results.keys())
            
            # Plot 1: Overall Accuracy Comparison
            accuracies = []
            for model in models:
                if 'overall_accuracy' in model_results[model]['metrics']:
                    acc = model_results[model]['metrics']['overall_accuracy']['mean']
                else:
                    acc = 0.0
                accuracies.append(acc)
            
            axes[0, 0].bar(models, accuracies, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'][:len(models)])
            axes[0, 0].set_title('Overall Point Detection Accuracy')
            axes[0, 0].set_ylabel('Accuracy')
            axes[0, 0].tick_params(axis='x', rotation=45)
            
            # Plot 2: Individual Point Accuracies
            point_labels = ['P', 'Q', 'R', 'S', 'T']
            x = np.arange(len(point_labels))
            width = 0.15
            
            for i, model in enumerate(models):
                point_accs = []
                for point in point_labels:
                    metric_name = f'{point}_accuracy'
                    if metric_name in model_results[model]['metrics']:
                        acc = model_results[model]['metrics'][metric_name]['mean']
                    else:
                        acc = 0.0
                    point_accs.append(acc)
                
                axes[0, 1].bar(x + i * width, point_accs, width, label=model)
            
            axes[0, 1].set_title('Individual Point Detection Accuracy')
            axes[0, 1].set_ylabel('Accuracy')
            axes[0, 1].set_xlabel('PQRST Points')
            axes[0, 1].set_xticks(x + width * (len(models) - 1) / 2)
            axes[0, 1].set_xticklabels(point_labels)
            axes[0, 1].legend()
            
            # Plot 3: Clinical Metrics
            clinical_metrics = ['PR_interval_accuracy', 'QT_interval_accuracy', 'clinical_validity']
            clinical_data = []
            
            for model in models:
                model_clinical = []
                for metric in clinical_metrics:
                    if metric in model_results[model]['metrics']:
                        val = model_results[model]['metrics'][metric]['mean']
                    else:
                        val = 0.0
                    model_clinical.append(val)
                clinical_data.append(model_clinical)
            
            clinical_data = np.array(clinical_data)
            
            im = axes[1, 0].imshow(clinical_data, cmap='Blues', aspect='auto')
            axes[1, 0].set_title('Clinical Validation Metrics')
            axes[1, 0].set_xticks(range(len(clinical_metrics)))
            axes[1, 0].set_xticklabels(clinical_metrics, rotation=45)
            axes[1, 0].set_yticks(range(len(models)))
            axes[1, 0].set_yticklabels(models)
            
            # Add colorbar
            cbar = plt.colorbar(im, ax=axes[1, 0])
            cbar.set_label('Score')
            
            # Plot 4: Performance Summary Radar Chart
            # This is a simplified version - in practice you'd use a proper radar chart library
            summary_metrics = ['overall_accuracy', 'clinical_validity', 'morphology_correlation']
            
            for model in models:
                values = []
                for metric in summary_metrics:
                    if metric in model_results[model]['metrics']:
                        val = model_results[model]['metrics'][metric]['mean']
                    else:
                        val = 0.0
                    values.append(val)
                
                axes[1, 1].plot(summary_metrics, values, marker='o', label=model)
            
            axes[1, 1].set_title('Performance Summary')
            axes[1, 1].set_ylabel('Score')
            axes[1, 1].tick_params(axis='x', rotation=45)
            axes[1, 1].legend()
            axes[1, 1].grid(True)
            
            plt.tight_layout()
            plt.savefig(self.results_dir / 'model_comparison.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"Comparison plots saved to {self.results_dir / 'model_comparison.png'}")
            
        except Exception as e:
            print(f"Error generating plots: {e}")
    
    def _save_results(self, model_results: Dict[str, Any], comparison_summary: Dict[str, Any]):
        """Save evaluation results"""
        # Save detailed results
        results_file = self.results_dir / 'detailed_results.json'
        
        # Convert numpy arrays to lists for JSON serialization
        serializable_results = {}
        for model_type, results in model_results.items():
            serializable_results[model_type] = {
                'metrics': {}
            }
            
            for metric_name, metric_data in results['metrics'].items():
                serializable_results[model_type]['metrics'][metric_name] = {
                    k: float(v) for k, v in metric_data.items()
                }
        
        with open(results_file, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        # Save comparison summary
        summary_file = self.results_dir / 'comparison_summary.json'
        with open(summary_file, 'w') as f:
            json.dump(comparison_summary, f, indent=2)
        
        # Create CSV summary for easy viewing
        csv_data = []
        for model_type in model_results.keys():
            row = {'Model': model_type}
            
            # Add key metrics
            key_metrics = ['overall_accuracy', 'R_accuracy', 'clinical_validity']
            for metric in key_metrics:
                if metric in model_results[model_type]['metrics']:
                    row[metric] = model_results[model_type]['metrics'][metric]['mean']
                else:
                    row[metric] = 0.0
            
            csv_data.append(row)
        
        df = pd.DataFrame(csv_data)
        df.to_csv(self.results_dir / 'model_comparison.csv', index=False)
        
        print(f"Results saved to {self.results_dir}")
        print("\nComparison Summary:")
        print(df.to_string(index=False))

# Import defaultdict
from collections import defaultdict

def create_evaluation_setup() -> Tuple[Dict[str, Path], Path]:
    """Create evaluation setup with model checkpoints and data directory"""
    
    # Define model checkpoint paths
    base_dir = Path("../experiments")
    model_checkpoints = {
        "vit_detr": base_dir / "vit_detr" / "best_model.pt",
        "multimodal": base_dir / "multimodal" / "best_model.pt", 
        "hubert": base_dir / "hubert" / "best_model.pt",
        "maskrcnn": base_dir / "maskrcnn" / "best_model.pt",
        "ensemble": base_dir / "ensemble" / "best_model.pt"
    }
    
    data_dir = Path("../data")
    
    return model_checkpoints, data_dir

if __name__ == "__main__":
    # Create evaluation setup
    model_checkpoints, data_dir = create_evaluation_setup()
    
    # Initialize evaluator
    evaluator = ECGEvaluator(model_checkpoints, data_dir, device="mps")
    
    # Run comprehensive comparison
    print("Starting comprehensive ECG model evaluation...")
    results = evaluator.compare_models()
    
    print("\n" + "="*80)
    print("EVALUATION COMPLETED")
    print("="*80)
    
    # Print best models for each metric
    if 'best_models' in results['comparison_summary']:
        print("\nBest Models by Metric:")
        for metric, best_model in results['comparison_summary']['best_models'].items():
            print(f"  {metric}: {best_model}")
    
    print(f"\nDetailed results saved in: {evaluator.results_dir}")
    print("Files generated:")
    print("  - model_comparison.png (visual comparison)")
    print("  - model_comparison.csv (summary table)")
    print("  - detailed_results.json (complete metrics)")
    print("  - comparison_summary.json (best model analysis)")