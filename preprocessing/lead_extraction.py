#!/usr/bin/env python3
"""
ECG Lead Extraction Module
Extracts individual ECG leads and traces using existing masks
"""

import cv2
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Optional
import json

class ECGLeadExtractor:
    """
    Extract individual ECG leads from scanned ECG images using mask guidance
    """
    
    def __init__(self, 
                 min_contour_area: int = 1000,
                 lead_names: List[str] = None):
        """
        Initialize ECG lead extractor
        
        Args:
            min_contour_area: Minimum area for valid lead contours
            lead_names: List of standard 12-lead ECG names
        """
        self.min_contour_area = min_contour_area
        self.lead_names = lead_names or [
            'I', 'II', 'III', 'aVR', 'aVL', 'aVF',
            'V1', 'V2', 'V3', 'V4', 'V5', 'V6'
        ]
    
    def extract_trace_regions(self, image: np.ndarray, mask: np.ndarray) -> List[Dict]:
        """
        Extract trace regions from ECG image using mask
        
        Args:
            image: Input ECG image (grayscale)
            mask: Binary mask highlighting ECG traces
            
        Returns:
            List of dictionaries containing trace region info
        """
        # Find contours in mask
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        trace_regions = []
        
        for i, contour in enumerate(contours):
            # Filter small contours
            area = cv2.contourArea(contour)
            if area < self.min_contour_area:
                continue
            
            # Get bounding rectangle
            x, y, w, h = cv2.boundingRect(contour)
            
            # Extract region from original image
            trace_region = image[y:y+h, x:x+w]
            
            # Create mask for this specific trace
            trace_mask = np.zeros_like(mask)
            cv2.drawContours(trace_mask, [contour], -1, 255, -1)
            region_mask = trace_mask[y:y+h, x:x+w]
            
            # Calculate trace statistics
            trace_info = {
                'id': i,
                'bbox': (x, y, w, h),
                'area': area,
                'trace_region': trace_region,
                'region_mask': region_mask,
                'contour': contour,
                'center': (x + w//2, y + h//2),
                'aspect_ratio': w / h if h > 0 else 0
            }
            
            trace_regions.append(trace_info)
        
        # Sort regions by position (top to bottom, left to right)
        trace_regions.sort(key=lambda x: (x['bbox'][1], x['bbox'][0]))
        
        return trace_regions
    
    def classify_lead_layout(self, trace_regions: List[Dict]) -> Dict[str, Dict]:
        """
        Classify trace regions into standard 12-lead layout
        
        Args:
            trace_regions: List of detected trace regions
            
        Returns:
            Dictionary mapping lead names to trace regions
        """
        if len(trace_regions) == 0:
            return {}
        
        # Group regions by vertical position (rows)
        y_positions = [region['bbox'][1] for region in trace_regions]
        y_sorted = sorted(set(y_positions))
        
        # Define row tolerance (regions in same row should be close vertically)
        row_tolerance = 50
        rows = []
        current_row = []
        current_y = y_sorted[0]
        
        for region in trace_regions:
            region_y = region['bbox'][1]
            
            if abs(region_y - current_y) <= row_tolerance:
                current_row.append(region)
            else:
                if current_row:
                    # Sort current row by x position
                    current_row.sort(key=lambda x: x['bbox'][0])
                    rows.append(current_row)
                current_row = [region]
                current_y = region_y
        
        # Add last row
        if current_row:
            current_row.sort(key=lambda x: x['bbox'][0])
            rows.append(current_row)
        
        # Map to lead names based on common ECG layouts
        lead_mapping = {}
        
        if len(rows) >= 3:
            # Standard 12-lead layout (usually 4 leads per row, 3 rows)
            lead_sequence = [
                ['I', 'aVR', 'V1', 'V4'],
                ['II', 'aVL', 'V2', 'V5'], 
                ['III', 'aVF', 'V3', 'V6']
            ]
            
            for row_idx, row_regions in enumerate(rows[:3]):
                for col_idx, region in enumerate(row_regions):
                    if row_idx < len(lead_sequence) and col_idx < len(lead_sequence[row_idx]):
                        lead_name = lead_sequence[row_idx][col_idx]
                        lead_mapping[lead_name] = region
        
        elif len(rows) == 2:
            # Alternative layout (6 leads per row)
            if len(rows[0]) >= 6:
                row1_leads = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF']
                row2_leads = ['V1', 'V2', 'V3', 'V4', 'V5', 'V6']
                
                for i, region in enumerate(rows[0][:6]):
                    lead_mapping[row1_leads[i]] = region
                
                for i, region in enumerate(rows[1][:6]):
                    lead_mapping[row2_leads[i]] = region
        
        else:
            # Sequential assignment for other layouts
            for i, region in enumerate(trace_regions):
                if i < len(self.lead_names):
                    lead_mapping[self.lead_names[i]] = region
        
        return lead_mapping
    
    def extract_lead_signals(self, lead_mapping: Dict[str, Dict]) -> Dict[str, np.ndarray]:
        """
        Extract 1D signal from each lead region
        
        Args:
            lead_mapping: Dictionary mapping lead names to regions
            
        Returns:
            Dictionary mapping lead names to 1D signals
        """
        lead_signals = {}
        
        for lead_name, region_info in lead_mapping.items():
            trace_region = region_info['trace_region']
            region_mask = region_info['region_mask']
            
            # Apply mask to trace region
            masked_trace = cv2.bitwise_and(trace_region, region_mask)
            
            # Extract signal by finding trace path
            signal = self._extract_trace_path(masked_trace, region_mask)
            
            lead_signals[lead_name] = signal
        
        return lead_signals
    
    def _extract_trace_path(self, trace_region: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """
        Extract 1D signal from trace region by following the trace path
        
        Args:
            trace_region: Image region containing the trace
            mask: Binary mask of the trace
            
        Returns:
            1D signal array
        """
        height, width = trace_region.shape
        
        # Method 1: Column-wise averaging
        signal = []
        
        for x in range(width):
            # Get column pixels where mask is active
            column_mask = mask[:, x]
            column_pixels = trace_region[:, x]
            
            if np.any(column_mask > 0):
                # Find trace pixels (dark pixels in the mask region)
                trace_pixels = column_pixels[column_mask > 0]
                
                # Use minimum value (darkest pixel) as trace point
                if len(trace_pixels) > 0:
                    trace_value = np.min(trace_pixels)
                    # Convert to y-coordinate (invert since ECG traces go up/down)
                    y_coord = height - (trace_value / 255.0) * height
                    signal.append(y_coord)
                else:
                    signal.append(height // 2)  # Default to middle
            else:
                signal.append(height // 2)  # Default to middle
        
        return np.array(signal)
    
    def process_ecg_image(self, image_path: str, mask_path: str) -> Dict:
        """
        Process a single ECG image and extract all leads
        
        Args:
            image_path: Path to ECG image
            mask_path: Path to corresponding mask
            
        Returns:
            Dictionary containing extracted leads and metadata
        """
        # Load image and mask
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
        if image is None or mask is None:
            raise ValueError(f"Could not load image or mask: {image_path}, {mask_path}")
        
        # Resize mask to match image if needed
        if image.shape != mask.shape:
            mask = cv2.resize(mask, (image.shape[1], image.shape[0]))
        
        # Threshold mask to ensure binary
        _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
        
        # Extract trace regions
        trace_regions = self.extract_trace_regions(image, mask)
        
        # Classify into leads
        lead_mapping = self.classify_lead_layout(trace_regions)
        
        # Extract signals
        lead_signals = self.extract_lead_signals(lead_mapping)
        
        # Prepare result
        result = {
            'image_path': image_path,
            'mask_path': mask_path,
            'image_shape': image.shape,
            'num_traces_detected': len(trace_regions),
            'leads_identified': list(lead_mapping.keys()),
            'lead_signals': lead_signals,
            'lead_mapping': {k: {
                'bbox': v['bbox'],
                'area': v['area'],
                'center': v['center'],
                'aspect_ratio': v['aspect_ratio']
            } for k, v in lead_mapping.items()},
            'trace_regions': [{
                'id': region['id'],
                'bbox': region['bbox'],
                'area': region['area'],
                'center': region['center'],
                'aspect_ratio': region['aspect_ratio']
            } for region in trace_regions]
        }
        
        return result
    
    def visualize_lead_extraction(self, image_path: str, mask_path: str, 
                                 output_path: str = None):
        """
        Visualize the lead extraction process
        
        Args:
            image_path: Path to ECG image
            mask_path: Path to mask
            output_path: Optional path to save visualization
        """
        # Process the image
        result = self.process_ecg_image(image_path, mask_path)
        
        # Load original images
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
        # Create visualization
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # Original image
        axes[0, 0].imshow(image, cmap='gray')
        axes[0, 0].set_title('Original ECG')
        axes[0, 0].axis('off')
        
        # Mask
        axes[0, 1].imshow(mask, cmap='gray')
        axes[0, 1].set_title('ECG Trace Mask')
        axes[0, 1].axis('off')
        
        # Detected traces with bounding boxes
        trace_vis = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), 
                 (255, 0, 255), (0, 255, 255)]
        
        for i, region in enumerate(result['trace_regions']):
            x, y, w, h = region['bbox']
            color = colors[i % len(colors)]
            cv2.rectangle(trace_vis, (x, y), (x+w, y+h), color, 2)
            cv2.putText(trace_vis, f"T{i}", (x, y-5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        axes[0, 2].imshow(trace_vis)
        axes[0, 2].set_title(f'Detected Traces ({len(result["trace_regions"])})')
        axes[0, 2].axis('off')
        
        # Lead mapping visualization
        lead_vis = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        for i, (lead_name, region_info) in enumerate(result['lead_mapping'].items()):
            x, y, w, h = region_info['bbox']
            color = colors[i % len(colors)]
            cv2.rectangle(lead_vis, (x, y), (x+w, y+h), color, 2)
            cv2.putText(lead_vis, lead_name, (x, y-5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        axes[1, 0].imshow(lead_vis)
        axes[1, 0].set_title(f'Lead Classification ({len(result["leads_identified"])} leads)')
        axes[1, 0].axis('off')
        
        # Sample extracted signals
        if result['lead_signals']:
            # Plot first few signals
            sample_leads = list(result['lead_signals'].keys())[:4]
            
            for i, lead_name in enumerate(sample_leads):
                signal = result['lead_signals'][lead_name]
                if i < 2:
                    axes[1, 1+i].plot(signal)
                    axes[1, 1+i].set_title(f'Lead {lead_name} Signal')
                    axes[1, 1+i].grid(True, alpha=0.3)
        else:
            axes[1, 1].text(0.5, 0.5, 'No signals extracted', 
                           ha='center', va='center', transform=axes[1, 1].transAxes)
            axes[1, 1].set_title('Signal Extraction')
            
            axes[1, 2].text(0.5, 0.5, 'No signals extracted', 
                           ha='center', va='center', transform=axes[1, 2].transAxes)
            axes[1, 2].set_title('Signal Extraction')
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"Visualization saved to: {output_path}")
        
        plt.show()
        
        # Print summary
        print(f"\nExtraction Summary:")
        print(f"Image: {Path(image_path).name}")
        print(f"Traces detected: {result['num_traces_detected']}")
        print(f"Leads identified: {', '.join(result['leads_identified'])}")
        print(f"Signals extracted: {len(result['lead_signals'])}")

def batch_extract_leads(images_dir: Path, masks_dir: Path, 
                       output_dir: Path) -> Dict:
    """
    Extract leads from a batch of ECG images
    
    Args:
        images_dir: Directory containing ECG images
        masks_dir: Directory containing masks
        output_dir: Directory to save results
        
    Returns:
        Batch processing statistics
    """
    images_dir = Path(images_dir)
    masks_dir = Path(masks_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    extractor = ECGLeadExtractor()
    
    stats = {
        "total_processed": 0,
        "total_leads_extracted": 0,
        "failed_files": [],
        "lead_statistics": {lead: 0 for lead in extractor.lead_names}
    }
    
    # Process all matching image-mask pairs
    for img_path in images_dir.glob("*.png"):
        mask_path = masks_dir / img_path.name
        
        if not mask_path.exists():
            print(f"No mask found for {img_path.name}")
            stats["failed_files"].append(str(img_path))
            continue
        
        try:
            # Extract leads
            result = extractor.process_ecg_image(str(img_path), str(mask_path))
            
            # Save results
            output_file = output_dir / f"{img_path.stem}_leads.json"
            
            # Convert numpy arrays to lists for JSON serialization
            json_result = result.copy()
            json_result['lead_signals'] = {
                k: v.tolist() for k, v in result['lead_signals'].items()
            }
            
            with open(output_file, 'w') as f:
                json.dump(json_result, f, indent=2)
            
            # Update statistics
            stats["total_processed"] += 1
            stats["total_leads_extracted"] += len(result['leads_identified'])
            
            for lead in result['leads_identified']:
                if lead in stats['lead_statistics']:
                    stats['lead_statistics'][lead] += 1
            
            print(f"Processed {img_path.name}: {len(result['leads_identified'])} leads extracted")
            
        except Exception as e:
            print(f"Failed to process {img_path.name}: {e}")
            stats["failed_files"].append(str(img_path))
    
    return stats

if __name__ == "__main__":
    # Example usage
    print("ECG Lead Extraction Module")
    
    # Test with sample data if available
    images_dir = Path("../data/raw/scanned_ecgs")
    masks_dir = Path("../data/annotations/manual")
    
    if images_dir.exists() and masks_dir.exists():
        sample_images = list(images_dir.glob("*.png"))[:3]
        
        if sample_images:
            extractor = ECGLeadExtractor()
            
            for img_path in sample_images:
                mask_path = masks_dir / img_path.name
                if mask_path.exists():
                    print(f"\nTesting with: {img_path.name}")
                    try:
                        extractor.visualize_lead_extraction(str(img_path), str(mask_path))
                    except Exception as e:
                        print(f"Error processing {img_path.name}: {e}")
                    break