#!/usr/bin/env python3
"""
Gemini AI 2.5 Flash ECG Analyzer
Integrates Google Gemini AI for intelligent ECG analysis and medical interpretation.
"""

import os
import json
import base64
from io import BytesIO
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import requests
from PIL import Image
import time

class GeminiECGAnalyzer:
    """Gemini AI interface for ECG analysis and medical interpretation."""
    
    def __init__(self, api_key: Optional[str] = None):
        """Initialize Gemini analyzer with API key."""
        self.api_key = api_key or os.getenv('GEMINI_API_KEY')
        if not self.api_key:
            raise ValueError("Gemini API key required. Set GEMINI_API_KEY environment variable or pass api_key parameter.")
        
        self.base_url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash-exp:generateContent"
        self.headers = {'Content-Type': 'application/json'}
        
        print("🤖 Gemini ECG Analyzer initialized")
        print(f"🔑 API Key: {'*' * 10}{self.api_key[-4:]}")
    
    def encode_image(self, image_path: str) -> str:
        """Encode image to base64 for Gemini API."""
        try:
            with open(image_path, 'rb') as image_file:
                encoded = base64.b64encode(image_file.read()).decode('utf-8')
            return encoded
        except Exception as e:
            raise Exception(f"Failed to encode image {image_path}: {e}")
    
    def create_ecg_analysis_prompt(self, detection_results: Dict, clinical_context: str = "") -> str:
        """Create specialized ECG analysis prompt for Gemini."""
        
        # Extract detection data
        num_detections = len(detection_results.get('confidences', []))
        max_confidence = max(detection_results.get('all_confidences', [0]))
        mean_confidence = sum(detection_results.get('all_confidences', [0])) / len(detection_results.get('all_confidences', [1]))
        
        # Calculate heart rate if we have detections
        heart_rate_estimate = "Unknown"
        if num_detections > 0:
            # Rough estimate assuming 10-second ECG strip
            heart_rate_estimate = f"~{num_detections * 6} BPM"
        
        prompt = f"""
You are an expert cardiologist analyzing an ECG image. Please provide a comprehensive medical analysis.

TECHNICAL DETECTION DATA:
- R-peaks detected: {num_detections}
- Maximum confidence: {max_confidence:.4f}
- Average confidence: {mean_confidence:.4f}  
- Estimated heart rate: {heart_rate_estimate}

CLINICAL ANALYSIS REQUEST:
1. RHYTHM ASSESSMENT:
   - Analyze the overall cardiac rhythm
   - Identify any arrhythmias or irregularities
   - Comment on heart rate and variability

2. WAVEFORM ANALYSIS:
   - Evaluate P-waves, QRS complexes, T-waves
   - Identify any abnormal morphology
   - Comment on intervals (PR, QT, QRS duration)

3. CLINICAL SIGNIFICANCE:
   - Assess overall cardiac health from this ECG
   - Identify any concerning findings
   - Rate clinical urgency (LOW/MODERATE/HIGH)

4. MEDICAL RECOMMENDATIONS:
   - Suggest appropriate follow-up care
   - Recommend additional testing if needed
   - Provide patient counseling points

5. RISK STRATIFICATION:
   - Overall cardiovascular risk level
   - Specific risk factors identified
   - Prognosis assessment

Please provide your analysis in a structured, professional medical format suitable for clinical documentation.

{clinical_context}
"""
        return prompt
    
    def analyze_ecg(self, image_path: str, detection_results: Dict, 
                   clinical_context: str = "") -> Dict:
        """
        Analyze ECG image and detection results with Gemini AI.
        
        Args:
            image_path: Path to ECG image
            detection_results: R-peak detection results from your model
            clinical_context: Additional clinical information
            
        Returns:
            Dictionary containing Gemini's medical analysis
        """
        try:
            print(f"🤖 Analyzing ECG with Gemini AI: {Path(image_path).name}")
            
            # Encode image
            image_b64 = self.encode_image(image_path)
            
            # Create analysis prompt
            analysis_prompt = self.create_ecg_analysis_prompt(detection_results, clinical_context)
            
            # Prepare API request
            payload = {
                "contents": [{
                    "parts": [
                        {
                            "text": analysis_prompt
                        },
                        {
                            "inline_data": {
                                "mime_type": "image/png",
                                "data": image_b64
                            }
                        }
                    ]
                }],
                "generationConfig": {
                    "temperature": 0.2,
                    "topK": 40,
                    "topP": 0.8,
                    "maxOutputTokens": 2048,
                }
            }
            
            # Make API request
            url = f"{self.base_url}?key={self.api_key}"
            response = requests.post(url, headers=self.headers, json=payload, timeout=30)
            
            if response.status_code == 200:
                result = response.json()
                
                # Extract analysis text
                analysis_text = ""
                if 'candidates' in result and len(result['candidates']) > 0:
                    if 'content' in result['candidates'][0]:
                        if 'parts' in result['candidates'][0]['content']:
                            analysis_text = result['candidates'][0]['content']['parts'][0].get('text', '')
                
                # Structure the response
                analysis_result = {
                    'status': 'success',
                    'image_analyzed': Path(image_path).name,
                    'detection_summary': {
                        'r_peaks_detected': len(detection_results.get('confidences', [])),
                        'max_confidence': float(max(detection_results.get('all_confidences', [0]))),
                        'mean_confidence': float(sum(detection_results.get('all_confidences', [0])) / len(detection_results.get('all_confidences', [1]))),
                    },
                    'gemini_analysis': analysis_text,
                    'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                    'api_model': 'gemini-2.0-flash-exp'
                }
                
                print(f"✅ Gemini analysis completed for {Path(image_path).name}")
                return analysis_result
                
            else:
                error_msg = f"Gemini API error {response.status_code}: {response.text}"
                print(f"❌ {error_msg}")
                return {
                    'status': 'error',
                    'error': error_msg,
                    'image_analyzed': Path(image_path).name
                }
                
        except Exception as e:
            error_msg = f"Gemini analysis failed: {str(e)}"
            print(f"❌ {error_msg}")
            return {
                'status': 'error',
                'error': error_msg,
                'image_analyzed': Path(image_path).name
            }
    
    def generate_clinical_report(self, analysis_result: Dict, patient_info: Dict = None) -> str:
        """Generate formatted clinical report from Gemini analysis."""
        
        if analysis_result.get('status') != 'success':
            return f"❌ Analysis failed: {analysis_result.get('error', 'Unknown error')}"
        
        # Header
        report = "🏥 COMPREHENSIVE ECG MEDICAL ANALYSIS REPORT\n"
        report += "=" * 60 + "\n"
        report += f"📁 Image: {analysis_result['image_analyzed']}\n"
        report += f"🤖 AI Analysis: Gemini 2.0 Flash\n"
        report += f"📅 Date: {analysis_result['timestamp']}\n"
        
        if patient_info:
            report += f"👤 Patient: {patient_info.get('name', 'N/A')}\n"
            report += f"🎂 Age: {patient_info.get('age', 'N/A')}\n"
        
        report += "\n"
        
        # Technical Summary
        detection = analysis_result['detection_summary']
        report += "📊 TECHNICAL DETECTION SUMMARY:\n"
        report += f"   R-peaks detected: {detection['r_peaks_detected']}\n"
        report += f"   Maximum confidence: {detection['max_confidence']:.4f}\n"
        report += f"   Average confidence: {detection['mean_confidence']:.4f}\n"
        report += f"   Estimated heart rate: ~{detection['r_peaks_detected'] * 6} BPM\n"
        report += "\n"
        
        # Gemini Medical Analysis
        report += "🤖 GEMINI AI MEDICAL ANALYSIS:\n"
        report += "-" * 40 + "\n"
        report += analysis_result['gemini_analysis']
        report += "\n\n"
        
        # Footer
        report += "=" * 60 + "\n"
        report += "🔬 Report generated by ECG-LLM system with Gemini AI integration\n"
        report += "⚠️  This analysis is for educational purposes. Consult a physician for medical decisions.\n"
        
        return report
    
    def batch_analyze_ecgs(self, ecg_results: List[Tuple[str, Dict]]) -> List[Dict]:
        """Analyze multiple ECG images and their detection results."""
        
        batch_results = []
        
        for i, (image_path, detection_results) in enumerate(ecg_results, 1):
            print(f"\n🔍 Processing ECG {i}/{len(ecg_results)}: {Path(image_path).name}")
            
            # Add delay to respect API limits
            if i > 1:
                time.sleep(1)
            
            analysis = self.analyze_ecg(image_path, detection_results)
            batch_results.append(analysis)
        
        print(f"\n✅ Batch analysis complete: {len(batch_results)} ECGs processed")
        return batch_results
    
    def save_analysis_report(self, analysis_result: Dict, output_path: str, 
                           patient_info: Dict = None):
        """Save Gemini analysis report to file."""
        
        # Generate clinical report
        report = self.generate_clinical_report(analysis_result, patient_info)
        
        # Save text report
        with open(output_path, 'w') as f:
            f.write(report)
        
        # Save JSON analysis
        json_path = output_path.replace('.txt', '.json')
        with open(json_path, 'w') as f:
            json.dump(analysis_result, f, indent=2)
        
        print(f"📄 Analysis report saved: {output_path}")
        print(f"📊 JSON data saved: {json_path}")


def test_gemini_connection(api_key: str = None) -> bool:
    """Test Gemini API connection with a simple request."""
    try:
        analyzer = GeminiECGAnalyzer(api_key)
        
        # Simple test request
        test_prompt = "Please respond with 'Gemini ECG Analyzer ready' to confirm connection."
        
        payload = {
            "contents": [{"parts": [{"text": test_prompt}]}],
            "generationConfig": {"temperature": 0, "maxOutputTokens": 50}
        }
        
        url = f"{analyzer.base_url}?key={analyzer.api_key}"
        response = requests.post(url, headers=analyzer.headers, json=payload, timeout=10)
        
        if response.status_code == 200:
            print("✅ Gemini API connection successful!")
            return True
        else:
            print(f"❌ Gemini API connection failed: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Gemini connection test failed: {e}")
        return False


if __name__ == "__main__":
    # Test the Gemini connection
    print("🧪 Testing Gemini ECG Analyzer...")
    
    # Test connection
    if test_gemini_connection():
        print("🎉 Gemini ECG Analyzer ready for integration!")
    else:
        print("⚠️  Please check your API key and try again.")