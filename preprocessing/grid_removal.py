#!/usr/bin/env python3
"""
ECG Grid Removal Module
Removes grid lines and background artifacts from scanned ECG images
"""

import cv2
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from typing import Tuple, Optional

class ECGGridRemover:
    """
    Advanced ECG grid removal using morphological operations and line detection
    """
    
    def __init__(self, 
                 grid_threshold: int = 30,
                 min_line_length: int = 50,
                 max_line_gap: int = 10,
                 morph_kernel_size: int = 3):
        """
        Initialize ECG grid remover
        
        Args:
            grid_threshold: Threshold for grid line detection
            min_line_length: Minimum length for detected lines
            max_line_gap: Maximum gap in line detection
            morph_kernel_size: Kernel size for morphological operations
        """
        self.grid_threshold = grid_threshold
        self.min_line_length = min_line_length
        self.max_line_gap = max_line_gap
        self.morph_kernel_size = morph_kernel_size
    
    def detect_grid_lines(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Detect horizontal and vertical grid lines using Hough transform
        
        Args:
            image: Input grayscale ECG image
            
        Returns:
            Tuple of (horizontal_lines, vertical_lines) coordinates
        """
        # Edge detection
        edges = cv2.Canny(image, 50, 150, apertureSize=3)
        
        # Detect lines using HoughLinesP
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, 
                               threshold=self.grid_threshold,
                               minLineLength=self.min_line_length,
                               maxLineGap=self.max_line_gap)
        
        horizontal_lines = []
        vertical_lines = []
        
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                
                # Calculate angle to determine if line is horizontal or vertical
                angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
                
                # Horizontal lines (within 15 degrees of horizontal)
                if abs(angle) < 15 or abs(angle) > 165:
                    horizontal_lines.append(line[0])
                
                # Vertical lines (within 15 degrees of vertical)
                elif 75 < abs(angle) < 105:
                    vertical_lines.append(line[0])
        
        return np.array(horizontal_lines), np.array(vertical_lines)
    
    def create_grid_mask(self, image: np.ndarray, 
                        horizontal_lines: np.ndarray, 
                        vertical_lines: np.ndarray,
                        line_thickness: int = 2) -> np.ndarray:
        """
        Create a binary mask for detected grid lines
        
        Args:
            image: Input image
            horizontal_lines: Array of horizontal line coordinates
            vertical_lines: Array of vertical line coordinates
            line_thickness: Thickness of lines in mask
            
        Returns:
            Binary mask where grid lines are white (255)
        """
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        
        # Draw horizontal lines
        if len(horizontal_lines) > 0:
            for line in horizontal_lines:
                x1, y1, x2, y2 = line
                cv2.line(mask, (x1, y1), (x2, y2), 255, line_thickness)
        
        # Draw vertical lines
        if len(vertical_lines) > 0:
            for line in vertical_lines:
                x1, y1, x2, y2 = line
                cv2.line(mask, (x1, y1), (x2, y2), 255, line_thickness)
        
        # Dilate mask to ensure complete line removal
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, 
                                         (self.morph_kernel_size, self.morph_kernel_size))
        mask = cv2.dilate(mask, kernel, iterations=1)
        
        return mask
    
    def remove_grid_morphological(self, image: np.ndarray) -> np.ndarray:
        """
        Remove grid using morphological operations (alternative method)
        
        Args:
            image: Input grayscale ECG image
            
        Returns:
            Image with grid removed using morphological operations
        """
        # Create kernels for horizontal and vertical line detection
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 1))
        vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 25))
        
        # Detect horizontal lines
        horizontal_lines = cv2.morphologyEx(image, cv2.MORPH_OPEN, horizontal_kernel)
        
        # Detect vertical lines
        vertical_lines = cv2.morphologyEx(image, cv2.MORPH_OPEN, vertical_kernel)
        
        # Combine detected lines
        grid_mask = cv2.add(horizontal_lines, vertical_lines)
        
        # Remove grid from original image
        result = cv2.subtract(image, grid_mask)
        
        return result
    
    def inpaint_grid_regions(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """
        Use image inpainting to fill grid line regions
        
        Args:
            image: Input image
            mask: Binary mask of grid regions to inpaint
            
        Returns:
            Image with grid regions inpainted
        """
        # Convert to 3-channel if grayscale
        if len(image.shape) == 2:
            image_3ch = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        else:
            image_3ch = image.copy()
        
        # Inpaint using Telea algorithm
        inpainted = cv2.inpaint(image_3ch, mask, 3, cv2.INPAINT_TELEA)
        
        # Convert back to grayscale if needed
        if len(image.shape) == 2:
            inpainted = cv2.cvtColor(inpainted, cv2.COLOR_BGR2GRAY)
        
        return inpainted
    
    def remove_grid(self, image: np.ndarray, method: str = "hough") -> Tuple[np.ndarray, dict]:
        """
        Main grid removal function with multiple methods
        
        Args:
            image: Input ECG image (grayscale or color)
            method: Grid removal method ("hough", "morphological", "combined")
            
        Returns:
            Tuple of (cleaned_image, processing_info)
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        processing_info = {
            "method": method,
            "original_shape": image.shape,
            "horizontal_lines_detected": 0,
            "vertical_lines_detected": 0
        }
        
        if method == "hough":
            # Hough transform method
            h_lines, v_lines = self.detect_grid_lines(gray)
            processing_info["horizontal_lines_detected"] = len(h_lines)
            processing_info["vertical_lines_detected"] = len(v_lines)
            
            # Create grid mask
            grid_mask = self.create_grid_mask(gray, h_lines, v_lines)
            
            # Inpaint grid regions
            cleaned = self.inpaint_grid_regions(gray, grid_mask)
            
        elif method == "morphological":
            # Morphological method
            cleaned = self.remove_grid_morphological(gray)
            
        elif method == "combined":
            # Combined approach
            # First apply morphological
            morph_cleaned = self.remove_grid_morphological(gray)
            
            # Then apply Hough on the result
            h_lines, v_lines = self.detect_grid_lines(morph_cleaned)
            processing_info["horizontal_lines_detected"] = len(h_lines)
            processing_info["vertical_lines_detected"] = len(v_lines)
            
            if len(h_lines) > 0 or len(v_lines) > 0:
                grid_mask = self.create_grid_mask(morph_cleaned, h_lines, v_lines)
                cleaned = self.inpaint_grid_regions(morph_cleaned, grid_mask)
            else:
                cleaned = morph_cleaned
        
        else:
            raise ValueError(f"Unknown method: {method}")
        
        # Enhance contrast
        cleaned = cv2.equalizeHist(cleaned)
        
        return cleaned, processing_info
    
    def process_batch(self, input_dir: Path, output_dir: Path, 
                     method: str = "combined") -> dict:
        """
        Process a batch of ECG images
        
        Args:
            input_dir: Directory containing input ECG images
            output_dir: Directory to save cleaned images
            method: Grid removal method
            
        Returns:
            Processing statistics
        """
        input_dir = Path(input_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        stats = {
            "total_processed": 0,
            "total_lines_removed": 0,
            "failed_files": []
        }
        
        # Process all PNG files
        for img_path in input_dir.glob("*.png"):
            try:
                # Load image
                image = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
                if image is None:
                    stats["failed_files"].append(str(img_path))
                    continue
                
                # Remove grid
                cleaned, info = self.remove_grid(image, method=method)
                
                # Save cleaned image
                output_path = output_dir / img_path.name
                cv2.imwrite(str(output_path), cleaned)
                
                # Update stats
                stats["total_processed"] += 1
                stats["total_lines_removed"] += (info["horizontal_lines_detected"] + 
                                                info["vertical_lines_detected"])
                
                print(f"Processed {img_path.name}: "
                      f"{info['horizontal_lines_detected']} H-lines, "
                      f"{info['vertical_lines_detected']} V-lines removed")
                
            except Exception as e:
                print(f"Failed to process {img_path.name}: {e}")
                stats["failed_files"].append(str(img_path))
        
        return stats

def visualize_grid_removal(image_path: str, output_path: str = None):
    """
    Visualize grid removal process for a single image
    
    Args:
        image_path: Path to input ECG image
        output_path: Optional path to save visualization
    """
    # Load image
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Could not load image: {image_path}")
    
    # Initialize grid remover
    grid_remover = ECGGridRemover()
    
    # Test different methods
    methods = ["morphological", "hough", "combined"]
    results = {}
    
    for method in methods:
        cleaned, info = grid_remover.remove_grid(image, method=method)
        results[method] = (cleaned, info)
    
    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Original image
    axes[0, 0].imshow(image, cmap='gray')
    axes[0, 0].set_title('Original ECG')
    axes[0, 0].axis('off')
    
    # Results for each method
    positions = [(0, 1), (1, 0), (1, 1)]
    for i, method in enumerate(methods):
        cleaned, info = results[method]
        row, col = positions[i]
        axes[row, col].imshow(cleaned, cmap='gray')
        axes[row, col].set_title(f'{method.capitalize()} Method\n'
                                f'H-lines: {info["horizontal_lines_detected"]}, '
                                f'V-lines: {info["vertical_lines_detected"]}')
        axes[row, col].axis('off')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Visualization saved to: {output_path}")
    
    plt.show()

if __name__ == "__main__":
    # Example usage
    print("ECG Grid Removal Module")
    print("Usage: python grid_removal.py")
    
    # Test with sample image if available
    sample_dir = Path("../data/raw/scanned_ecgs")
    if sample_dir.exists():
        sample_images = list(sample_dir.glob("*.png"))
        if sample_images:
            print(f"Testing with sample image: {sample_images[0]}")
            visualize_grid_removal(str(sample_images[0]))