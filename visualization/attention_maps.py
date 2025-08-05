#!/usr/bin/env python3
"""
Attention Visualization for ECG Models
Provides interpretable attention maps and model explanations for ECG analysis.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
from PIL import Image
import cv2
from typing import Dict, List, Tuple, Optional, Union
from pathlib import Path
import json

# For gradient-based methods
from captum.attr import IntegratedGradients, GradCAM, GuidedGradCam, Occlusion
from captum.attr import visualization as viz


class AttentionVisualizer:
    """Comprehensive attention visualization for ECG models."""
    
    def __init__(self, model: nn.Module, device: str = 'mps'):
        """
        Initialize attention visualizer.
        
        Args:
            model: The ECG model to visualize
            device: Device to run computations on
        """
        self.model = model.to(device)
        self.device = device
        self.model.eval()
        
        # Hook storage for intermediate activations
        self.attention_maps = {}
        self.feature_maps = {}
        self.hooks = []
        
        # Register hooks for attention extraction
        self._register_attention_hooks()
        
    def _register_attention_hooks(self):
        """Register forward hooks to capture attention maps."""
        def attention_hook(name):
            def hook(module, input, output):
                if hasattr(output, 'attentions') and output.attentions is not None:
                    self.attention_maps[name] = output.attentions.detach().cpu()
                elif isinstance(output, tuple) and len(output) > 1:
                    # Handle transformer outputs
                    if hasattr(output[0], 'attentions'):
                        self.attention_maps[name] = output[0].attentions.detach().cpu()
            return hook
        
        def feature_hook(name):
            def hook(module, input, output):
                if isinstance(output, torch.Tensor):
                    self.feature_maps[name] = output.detach().cpu()
            return hook
        
        # Register hooks based on model architecture
        if hasattr(self.model, 'vit_backbone'):
            # ViT-based model
            for i, layer in enumerate(self.model.vit_backbone.encoder.layer):
                hook = layer.attention.register_forward_hook(attention_hook(f'vit_layer_{i}'))
                self.hooks.append(hook)
                
        if hasattr(self.model, 'decoder'):
            # DETR decoder
            for i, layer in enumerate(self.model.decoder.transformer_decoder.layers):
                hook = layer.self_attn.register_forward_hook(attention_hook(f'decoder_layer_{i}'))
                self.hooks.append(hook)
    
    def get_attention_maps(self, image: torch.Tensor, layer_idx: Optional[int] = None) -> Dict[str, torch.Tensor]:
        """
        Extract attention maps from model.
        
        Args:
            image: Input ECG image tensor [1, C, H, W]
            layer_idx: Specific layer to extract (None for all layers)
            
        Returns:
            Dictionary of attention maps
        """
        self.attention_maps.clear()
        self.feature_maps.clear()
        
        with torch.no_grad():
            image = image.to(self.device)
            _ = self.model(image)
        
        if layer_idx is not None:
            layer_name = f'vit_layer_{layer_idx}'
            return {layer_name: self.attention_maps.get(layer_name, None)}
        
        return self.attention_maps.copy()
    
    def visualize_attention_maps(self, 
                               image: torch.Tensor, 
                               original_image: np.ndarray,
                               layer_indices: List[int] = [0, 5, 11],
                               head_indices: List[int] = [0, 5, 11],
                               save_path: Optional[str] = None) -> plt.Figure:
        """
        Create comprehensive attention visualization.
        
        Args:
            image: Preprocessed image tensor
            original_image: Original ECG image as numpy array
            layer_indices: Which transformer layers to visualize
            head_indices: Which attention heads to visualize
            save_path: Path to save the visualization
            
        Returns:
            Matplotlib figure
        """
        attention_maps = self.get_attention_maps(image)
        
        # Create figure with subplots
        fig, axes = plt.subplots(len(layer_indices), len(head_indices) + 1, 
                               figsize=(20, 15))
        fig.suptitle('ECG Attention Map Visualization', fontsize=16, fontweight='bold')
        
        # Original image
        for i in range(len(layer_indices)):
            axes[i, 0].imshow(original_image)
            axes[i, 0].set_title(f'Original ECG\n(Layer {layer_indices[i]})', fontweight='bold')
            axes[i, 0].axis('off')
        
        # Process each layer and head
        for layer_idx, layer_num in enumerate(layer_indices):
            layer_name = f'vit_layer_{layer_num}'
            
            if layer_name not in attention_maps:
                continue
                
            attn = attention_maps[layer_name][0]  # Remove batch dimension
            
            for head_idx, head_num in enumerate(head_indices):
                if head_num >= attn.shape[0]:
                    continue
                
                # Get attention for this head
                head_attn = attn[head_num]  # [seq_len, seq_len]
                
                # Average attention from CLS token to all patches
                cls_attn = head_attn[0, 1:]  # Remove CLS->CLS attention
                
                # Reshape to spatial dimensions
                patch_size = int(np.sqrt(len(cls_attn)))
                if patch_size * patch_size == len(cls_attn):
                    attn_map = cls_attn.reshape(patch_size, patch_size)
                    
                    # Resize to match original image dimensions
                    attn_map = cv2.resize(
                        attn_map.numpy(), 
                        (original_image.shape[1], original_image.shape[0])
                    )
                    
                    # Create overlay
                    ax = axes[layer_idx, head_idx + 1]
                    ax.imshow(original_image, alpha=0.6)
                    
                    # Apply attention heatmap
                    im = ax.imshow(attn_map, cmap='hot', alpha=0.7, 
                                 vmin=attn_map.min(), vmax=attn_map.max())
                    
                    ax.set_title(f'Layer {layer_num}, Head {head_num}', fontweight='bold')
                    ax.axis('off')
                    
                    # Add colorbar
                    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def get_attention_rollout(self, image: torch.Tensor, discard_ratio: float = 0.9) -> np.ndarray:
        """
        Compute attention rollout for global model understanding.
        
        Args:
            image: Input image tensor
            discard_ratio: Ratio of attention to discard (keep top percentage)
            
        Returns:
            Attention rollout map
        """
        attention_maps = self.get_attention_maps(image)
        
        # Collect all attention matrices
        all_attentions = []
        for layer_name in sorted(attention_maps.keys()):
            if 'vit_layer' in layer_name:
                attn = attention_maps[layer_name][0]  # Remove batch dimension
                # Average over all heads
                attn = attn.mean(dim=0)
                all_attentions.append(attn)
        
        if not all_attentions:
            return np.zeros((224, 224))
        
        # Apply attention rollout
        result = torch.eye(all_attentions[0].shape[0])
        
        for attn in all_attentions:
            # Apply residual connection
            attn = attn + torch.eye(attn.shape[0])
            attn = attn / attn.sum(dim=-1, keepdim=True)
            
            # Apply discard ratio
            flat_attn = attn.flatten()
            _, indices = torch.topk(flat_attn, int(flat_attn.shape[0] * discard_ratio))
            flat_attn[indices] = 0
            attn = flat_attn.reshape(attn.shape)
            attn = attn / attn.sum(dim=-1, keepdim=True)
            
            result = torch.matmul(attn, result)
        
        # Extract CLS attention to patches
        rollout = result[0, 1:]
        
        # Reshape to spatial dimensions
        patch_size = int(np.sqrt(len(rollout)))
        if patch_size * patch_size == len(rollout):
            rollout_map = rollout.reshape(patch_size, patch_size)
            return rollout_map.numpy()
        
        return np.zeros((patch_size, patch_size))
    
    def grad_cam_visualization(self, 
                             image: torch.Tensor, 
                             original_image: np.ndarray,
                             target_class: int = None,
                             save_path: Optional[str] = None) -> plt.Figure:
        """
        Generate Grad-CAM visualization for model interpretability.
        
        Args:
            image: Preprocessed image tensor
            original_image: Original ECG image
            target_class: Target class for Grad-CAM (None for predicted class)
            save_path: Path to save visualization
            
        Returns:
            Matplotlib figure
        """
        # Get model's final convolutional layer
        target_layer = None
        if hasattr(self.model, 'vit_backbone'):
            # For ViT, use the last layer before classification
            target_layer = self.model.vit_backbone.encoder.layer[-1].output
        
        if target_layer is None:
            print("Could not find suitable layer for Grad-CAM")
            return None
        
        # Initialize Grad-CAM
        grad_cam = GradCAM(self.model, target_layer)
        
        # Prepare input
        input_tensor = image.to(self.device)
        if input_tensor.dim() == 3:
            input_tensor = input_tensor.unsqueeze(0)
        
        # Get target class
        if target_class is None:
            with torch.no_grad():
                output = self.model(input_tensor)
                if isinstance(output, dict):
                    # Handle DETR-style output
                    class_logits = output.get('class_logits')
                    if class_logits is not None:
                        target_class = class_logits.argmax(dim=-1)[0, 0].item()
                else:
                    target_class = output.argmax(dim=1)[0].item()
        
        # Generate attribution
        try:
            attribution = grad_cam.attribute(input_tensor, target=target_class)
            attribution = attribution.squeeze().cpu().numpy()
            
            # Create visualization
            fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
            
            # Original image
            ax1.imshow(original_image)
            ax1.set_title('Original ECG')
            ax1.axis('off')
            
            # Grad-CAM heatmap
            ax2.imshow(attribution, cmap='hot')
            ax2.set_title(f'Grad-CAM (Class {target_class})')
            ax2.axis('off')
            
            # Overlay
            ax3.imshow(original_image, alpha=0.6)
            ax3.imshow(attribution, cmap='hot', alpha=0.4)
            ax3.set_title('Overlay')
            ax3.axis('off')
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
            return fig
            
        except Exception as e:
            print(f"Error generating Grad-CAM: {e}")
            return None
    
    def integrated_gradients_visualization(self,
                                         image: torch.Tensor,
                                         original_image: np.ndarray,
                                         target_class: int = None,
                                         n_steps: int = 200,
                                         save_path: Optional[str] = None) -> plt.Figure:
        """
        Generate Integrated Gradients visualization.
        
        Args:
            image: Preprocessed image tensor
            original_image: Original ECG image
            target_class: Target class (None for predicted)
            n_steps: Number of integration steps
            save_path: Path to save visualization
            
        Returns:
            Matplotlib figure
        """
        # Initialize Integrated Gradients
        ig = IntegratedGradients(self.model)
        
        # Prepare input
        input_tensor = image.to(self.device)
        if input_tensor.dim() == 3:
            input_tensor = input_tensor.unsqueeze(0)
        
        # Create baseline (black image)
        baseline = torch.zeros_like(input_tensor)
        
        # Get target class
        if target_class is None:
            with torch.no_grad():
                output = self.model(input_tensor)
                if isinstance(output, dict):
                    class_logits = output.get('class_logits')
                    if class_logits is not None:
                        target_class = class_logits.argmax(dim=-1)[0, 0].item()
                else:
                    target_class = output.argmax(dim=1)[0].item()
        
        try:
            # Generate attribution
            attribution = ig.attribute(
                input_tensor,
                baselines=baseline,
                target=target_class,
                n_steps=n_steps
            )
            
            # Process attribution for visualization
            attribution = attribution.squeeze().cpu().numpy()
            attribution = np.transpose(attribution, (1, 2, 0))
            
            # Create visualization
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            fig.suptitle('Integrated Gradients Analysis', fontsize=16)
            
            # Original image
            axes[0, 0].imshow(original_image)
            axes[0, 0].set_title('Original ECG')
            axes[0, 0].axis('off')
            
            # Attribution heatmap
            attr_gray = np.mean(np.abs(attribution), axis=2)
            im1 = axes[0, 1].imshow(attr_gray, cmap='hot')
            axes[0, 1].set_title(f'Attribution Magnitude (Class {target_class})')
            axes[0, 1].axis('off')
            plt.colorbar(im1, ax=axes[0, 1])
            
            # Positive attribution
            attr_pos = np.maximum(attribution, 0)
            attr_pos = np.mean(attr_pos, axis=2)
            im2 = axes[1, 0].imshow(attr_pos, cmap='Reds')
            axes[1, 0].set_title('Positive Attribution')
            axes[1, 0].axis('off')
            plt.colorbar(im2, ax=axes[1, 0])
            
            # Negative attribution
            attr_neg = np.minimum(attribution, 0)
            attr_neg = np.abs(np.mean(attr_neg, axis=2))
            im3 = axes[1, 1].imshow(attr_neg, cmap='Blues')
            axes[1, 1].set_title('Negative Attribution')
            axes[1, 1].axis('off')
            plt.colorbar(im3, ax=axes[1, 1])
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
            return fig
            
        except Exception as e:
            print(f"Error generating Integrated Gradients: {e}")
            return None
    
    def occlusion_sensitivity(self,
                            image: torch.Tensor,
                            original_image: np.ndarray,
                            target_class: int = None,
                            sliding_window_shapes: Tuple[int, int] = (15, 15),
                            strides: Tuple[int, int] = (8, 8),
                            save_path: Optional[str] = None) -> plt.Figure:
        """
        Generate occlusion sensitivity map.
        
        Args:
            image: Preprocessed image tensor
            original_image: Original ECG image
            target_class: Target class
            sliding_window_shapes: Size of occlusion window
            strides: Stride for sliding window
            save_path: Path to save visualization
            
        Returns:
            Matplotlib figure
        """
        # Initialize Occlusion
        occlusion = Occlusion(self.model)
        
        # Prepare input
        input_tensor = image.to(self.device)
        if input_tensor.dim() == 3:
            input_tensor = input_tensor.unsqueeze(0)
        
        # Get target class
        if target_class is None:
            with torch.no_grad():
                output = self.model(input_tensor)
                if isinstance(output, dict):
                    class_logits = output.get('class_logits')
                    if class_logits is not None:
                        target_class = class_logits.argmax(dim=-1)[0, 0].item()
                else:
                    target_class = output.argmax(dim=1)[0].item()
        
        try:
            # Generate occlusion sensitivity
            attribution = occlusion.attribute(
                input_tensor,
                strides=strides,
                target=target_class,
                sliding_window_shapes=sliding_window_shapes,
                baselines=0
            )
            
            attribution = attribution.squeeze().cpu().numpy()
            attribution = np.mean(attribution, axis=0)  # Average over channels
            
            # Create visualization
            fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
            fig.suptitle('Occlusion Sensitivity Analysis', fontsize=16)
            
            # Original image
            ax1.imshow(original_image)
            ax1.set_title('Original ECG')
            ax1.axis('off')
            
            # Occlusion sensitivity map
            im = ax2.imshow(attribution, cmap='RdBu_r')
            ax2.set_title(f'Sensitivity Map (Class {target_class})')
            ax2.axis('off')
            plt.colorbar(im, ax=ax2)
            
            # Overlay
            ax3.imshow(original_image, alpha=0.7)
            ax3.imshow(attribution, cmap='RdBu_r', alpha=0.5)
            ax3.set_title('Overlay')
            ax3.axis('off')
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
            return fig
            
        except Exception as e:
            print(f"Error generating occlusion sensitivity: {e}")
            return None
    
    def generate_comprehensive_report(self,
                                    image: torch.Tensor,
                                    original_image: np.ndarray,
                                    image_id: str,
                                    output_dir: str) -> Dict[str, str]:
        """
        Generate comprehensive interpretability report.
        
        Args:
            image: Preprocessed image tensor
            original_image: Original ECG image
            image_id: Unique identifier for the image
            output_dir: Directory to save all visualizations
            
        Returns:
            Dictionary with paths to generated visualizations
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        report_paths = {}
        
        # Attention maps
        print("Generating attention maps...")
        attn_fig = self.visualize_attention_maps(
            image, original_image,
            save_path=str(output_path / f"{image_id}_attention_maps.png")
        )
        report_paths['attention_maps'] = f"{image_id}_attention_maps.png"
        plt.close(attn_fig)
        
        # Grad-CAM
        print("Generating Grad-CAM...")
        gradcam_fig = self.grad_cam_visualization(
            image, original_image,
            save_path=str(output_path / f"{image_id}_gradcam.png")
        )
        if gradcam_fig:
            report_paths['gradcam'] = f"{image_id}_gradcam.png"
            plt.close(gradcam_fig)
        
        # Integrated Gradients
        print("Generating Integrated Gradients...")
        ig_fig = self.integrated_gradients_visualization(
            image, original_image,
            save_path=str(output_path / f"{image_id}_integrated_gradients.png")
        )
        if ig_fig:
            report_paths['integrated_gradients'] = f"{image_id}_integrated_gradients.png"
            plt.close(ig_fig)
        
        # Occlusion Sensitivity
        print("Generating Occlusion Sensitivity...")
        occ_fig = self.occlusion_sensitivity(
            image, original_image,
            save_path=str(output_path / f"{image_id}_occlusion.png")
        )
        if occ_fig:
            report_paths['occlusion'] = f"{image_id}_occlusion.png"
            plt.close(occ_fig)
        
        # Save report metadata
        report_data = {
            'image_id': image_id,
            'generated_files': report_paths,
            'model_info': {
                'architecture': type(self.model).__name__,
                'device': self.device
            }
        }
        
        with open(output_path / f"{image_id}_report.json", 'w') as f:
            json.dump(report_data, f, indent=2)
        
        report_paths['metadata'] = f"{image_id}_report.json"
        
        print(f"✅ Comprehensive report generated for {image_id}")
        return report_paths
    
    def cleanup(self):
        """Remove registered hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()


# Example usage and testing
if __name__ == "__main__":
    # This would typically be used with a trained model
    print("Attention Visualization module loaded successfully!")
    print("Usage example:")
    print("""
    from models.backbones.vision_transformer_ecg import create_ecg_vit_detr_model
    from visualization.attention_maps import AttentionVisualizer
    
    # Load model
    model = create_ecg_vit_detr_model(pretrained=True)
    
    # Initialize visualizer  
    visualizer = AttentionVisualizer(model, device='mps')
    
    # Generate visualizations
    image_tensor = preprocess_ecg_image(image_path)
    original_image = load_original_image(image_path)
    
    # Create comprehensive report
    report_paths = visualizer.generate_comprehensive_report(
        image_tensor, original_image, 'ecg_001', './reports'
    )
    """)