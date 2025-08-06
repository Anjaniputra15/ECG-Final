#!/usr/bin/env python3
"""
ECG Result Analyzer Bridge
Connects R-peak detection results with Gemini AI analysis for comprehensive ECG interpretation.
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
from datetime import datetime

# Add current directory to path for imports
current_dir = Path(__file__).parent.parent
sys.path.append(str(current_dir))

from AIanalysis.gemini.gemini_analyzer import GeminiECGAnalyzer


class ECGResultAnalyzer:
    """Bridge between ECG detection results and Gemini AI analysis."""
    
    def __init__(self, gemini_api_key: Optional[str] = None):
        """Initialize ECG result analyzer with Gemini integration."""
        self.gemini_analyzer = GeminiECGAnalyzer(gemini_api_key) if gemini_api_key else None
        self.analysis_history = []
        
        print("🔗 ECG Result Analyzer Bridge initialized")
        if self.gemini_analyzer:
            print("✅ Gemini AI integration enabled")
        else:
            print("⚠️  Gemini AI disabled - provide API key to enable")
    
    def analyze_detection_results(self, detection_results: Dict) -> Dict:
        """Analyze R-peak detection results and extract medical insights."""
        
        # Basic metrics from detection results
        all_confidences = detection_results.get('all_confidences', [])
        detected_peaks = detection_results.get('confidences', [])
        keypoints = detection_results.get('keypoints', [])
        
        # Calculate heart rate metrics
        num_detections = len(detected_peaks)
        estimated_hr = num_detections * 6  # Assuming 10-second ECG strip
        
        # Confidence analysis
        max_confidence = float(np.max(all_confidences)) if len(all_confidences) > 0 else 0.0
        mean_confidence = float(np.mean(all_confidences)) if len(all_confidences) > 0 else 0.0
        confidence_std = float(np.std(all_confidences)) if len(all_confidences) > 0 else 0.0
        
        # R-peak interval analysis (if we have multiple peaks)
        rr_intervals = []
        if len(keypoints) > 1:
            for i in range(1, len(keypoints)):
                # Calculate distance between consecutive R-peaks (normalized coordinates)
                prev_x = keypoints[i-1][0]
                curr_x = keypoints[i][0]
                rr_interval = abs(curr_x - prev_x)
                rr_intervals.append(rr_interval)
        
        # Heart rate variability metrics
        hrv_metrics = {}
        if len(rr_intervals) > 0:
            hrv_metrics = {
                'mean_rr_interval': float(np.mean(rr_intervals)),
                'rr_std': float(np.std(rr_intervals)),
                'rr_variability_coefficient': float(np.std(rr_intervals) / np.mean(rr_intervals)) if np.mean(rr_intervals) > 0 else 0,
                'num_intervals': len(rr_intervals)
            }
        
        # Clinical interpretation
        clinical_assessment = self._assess_clinical_significance(
            estimated_hr, max_confidence, mean_confidence, confidence_std, hrv_metrics
        )
        
        analysis = {
            'timestamp': datetime.now().isoformat(),
            'detection_summary': {
                'total_detections': num_detections,
                'high_confidence_detections': len(detected_peaks),
                'estimated_heart_rate_bpm': estimated_hr,
                'max_confidence': max_confidence,
                'mean_confidence': mean_confidence,
                'confidence_std': confidence_std
            },
            'heart_rate_analysis': {
                'estimated_bpm': estimated_hr,
                'hr_category': self._categorize_heart_rate(estimated_hr),
                'hr_assessment': self._assess_heart_rate(estimated_hr)
            },
            'rhythm_analysis': {
                'rr_intervals': rr_intervals,
                'hrv_metrics': hrv_metrics,
                'rhythm_regularity': self._assess_rhythm_regularity(rr_intervals)
            },
            'confidence_analysis': {
                'detection_quality': self._assess_detection_quality(mean_confidence),
                'confidence_interpretation': self._interpret_confidence_levels(mean_confidence, confidence_std)
            },
            'clinical_assessment': clinical_assessment
        }
        
        return analysis
    
    def _categorize_heart_rate(self, hr_bpm: int) -> str:
        """Categorize heart rate into clinical ranges."""
        if hr_bpm < 60:
            return "Bradycardia"
        elif hr_bpm > 100:
            return "Tachycardia"
        else:
            return "Normal"
    
    def _assess_heart_rate(self, hr_bpm: int) -> str:
        """Provide clinical assessment of heart rate."""
        if hr_bpm < 50:
            return "Severe bradycardia - immediate medical attention recommended"
        elif hr_bpm < 60:
            return "Mild bradycardia - monitor and evaluate underlying causes"
        elif hr_bpm <= 100:
            return "Normal heart rate range"
        elif hr_bpm <= 150:
            return "Mild tachycardia - evaluate for underlying causes"
        else:
            return "Severe tachycardia - immediate medical evaluation required"
    
    def _assess_rhythm_regularity(self, rr_intervals: List[float]) -> Dict:
        """Assess rhythm regularity from RR intervals."""
        if len(rr_intervals) < 2:
            return {'assessment': 'Insufficient data', 'regularity_score': 0}
        
        rr_std = np.std(rr_intervals)
        rr_mean = np.mean(rr_intervals)
        
        variability_coefficient = rr_std / rr_mean if rr_mean > 0 else float('inf')
        
        if variability_coefficient < 0.1:
            regularity = "Very regular rhythm"
            score = 0.9
        elif variability_coefficient < 0.2:
            regularity = "Regular rhythm with minor variations"
            score = 0.7
        elif variability_coefficient < 0.4:
            regularity = "Irregular rhythm"
            score = 0.4
        else:
            regularity = "Highly irregular rhythm - possible arrhythmia"
            score = 0.1
        
        return {
            'assessment': regularity,
            'variability_coefficient': float(variability_coefficient),
            'regularity_score': score,
            'rr_std': float(rr_std),
            'rr_mean': float(rr_mean)
        }
    
    def _assess_detection_quality(self, mean_confidence: float) -> str:
        """Assess the quality of R-peak detection."""
        if mean_confidence > 0.8:
            return "Excellent detection quality"
        elif mean_confidence > 0.6:
            return "Good detection quality"
        elif mean_confidence > 0.4:
            return "Fair detection quality"
        elif mean_confidence > 0.2:
            return "Poor detection quality"
        else:
            return "Very poor detection quality - results may be unreliable"
    
    def _interpret_confidence_levels(self, mean_conf: float, std_conf: float) -> str:
        """Interpret confidence levels for clinical context."""
        if mean_conf > 0.5:
            return "High confidence detections - results likely reliable"
        elif mean_conf > 0.2:
            if std_conf < 0.1:
                return "Moderate confidence with consistent detections"
            else:
                return "Moderate confidence with variable detection quality"
        else:
            return "Low confidence detections - manual review recommended"
    
    def _assess_clinical_significance(self, hr_bpm: int, max_conf: float, 
                                   mean_conf: float, conf_std: float, 
                                   hrv_metrics: Dict) -> Dict:
        """Provide overall clinical significance assessment."""
        
        # Risk level calculation
        risk_factors = []
        risk_level = "LOW"
        
        # Heart rate risk factors
        if hr_bpm < 50 or hr_bpm > 150:
            risk_factors.append("Abnormal heart rate")
            risk_level = "HIGH"
        elif hr_bpm < 60 or hr_bpm > 100:
            risk_factors.append("Heart rate outside normal range")
            if risk_level == "LOW":
                risk_level = "MODERATE"
        
        # Detection quality risk factors
        if mean_conf < 0.2:
            risk_factors.append("Poor detection quality")
            if risk_level == "LOW":
                risk_level = "MODERATE"
        
        # Rhythm irregularity risk factors
        if hrv_metrics and hrv_metrics.get('rr_variability_coefficient', 0) > 0.4:
            risk_factors.append("Irregular rhythm detected")
            if risk_level != "HIGH":
                risk_level = "MODERATE"
        
        # Recommendations
        recommendations = []
        if risk_level == "HIGH":
            recommendations.append("Immediate medical evaluation recommended")
            recommendations.append("Consider continuous cardiac monitoring")
        elif risk_level == "MODERATE":
            recommendations.append("Follow-up with healthcare provider")
            recommendations.append("Consider repeat ECG")
        else:
            recommendations.append("Routine monitoring sufficient")
        
        if mean_conf < 0.3:
            recommendations.append("Manual ECG review recommended due to low detection confidence")
        
        return {
            'risk_level': risk_level,
            'risk_factors': risk_factors,
            'recommendations': recommendations,
            'urgency': 'IMMEDIATE' if risk_level == "HIGH" else 'ROUTINE',
            'follow_up_needed': risk_level in ["HIGH", "MODERATE"]
        }
    
    def analyze_with_gemini(self, image_path: str, detection_results: Dict,
                           patient_info: Dict = None) -> Dict:
        """Complete analysis combining technical results with Gemini AI insights."""
        
        if not self.gemini_analyzer:
            return {
                'status': 'error',
                'error': 'Gemini AI not available - API key required'
            }
        
        print(f"🔍 Analyzing ECG: {Path(image_path).name}")
        
        # 1. Technical analysis of detection results
        technical_analysis = self.analyze_detection_results(detection_results)
        
        # 2. Gemini AI medical analysis
        gemini_analysis = self.gemini_analyzer.analyze_ecg(
            image_path, detection_results, 
            clinical_context=f"Patient info: {patient_info}" if patient_info else ""
        )
        
        # 3. Combined comprehensive analysis
        comprehensive_analysis = {
            'status': 'success',
            'image_path': image_path,
            'timestamp': datetime.now().isoformat(),
            'technical_analysis': technical_analysis,
            'gemini_analysis': gemini_analysis,
            'patient_info': patient_info or {},
            'combined_assessment': self._create_combined_assessment(
                technical_analysis, gemini_analysis
            )
        }
        
        # Store in history
        self.analysis_history.append(comprehensive_analysis)
        
        return comprehensive_analysis
    
    def _create_combined_assessment(self, technical: Dict, gemini: Dict) -> Dict:
        """Create combined assessment from technical and AI analysis."""
        
        # Extract key findings
        tech_risk = technical['clinical_assessment']['risk_level']
        tech_hr = technical['heart_rate_analysis']['estimated_bpm']
        tech_quality = technical['confidence_analysis']['detection_quality']
        
        gemini_success = gemini.get('status') == 'success'
        
        combined = {
            'overall_risk_level': tech_risk,
            'heart_rate_assessment': f"{tech_hr} BPM - {technical['heart_rate_analysis']['hr_category']}",
            'detection_reliability': tech_quality,
            'ai_analysis_available': gemini_success,
            'recommendations': technical['clinical_assessment']['recommendations']
        }
        
        if gemini_success and 'gemini_analysis' in gemini:
            combined['ai_insights'] = "Gemini AI medical analysis completed"
            combined['ai_enhanced'] = True
        else:
            combined['ai_insights'] = "Technical analysis only"
            combined['ai_enhanced'] = False
        
        return combined
    
    def generate_summary_report(self, comprehensive_analysis: Dict) -> str:
        """Generate human-readable summary report."""
        
        if comprehensive_analysis.get('status') != 'success':
            return f"❌ Analysis failed: {comprehensive_analysis.get('error', 'Unknown error')}"
        
        tech = comprehensive_analysis['technical_analysis']
        combined = comprehensive_analysis['combined_assessment']
        
        report = f"""
🏥 ECG ANALYSIS SUMMARY REPORT
{'='*50}
📁 Image: {Path(comprehensive_analysis['image_path']).name}
📅 Analysis Date: {comprehensive_analysis['timestamp'][:19]}

📊 TECHNICAL FINDINGS:
   Heart Rate: {combined['heart_rate_assessment']}
   R-peaks Detected: {tech['detection_summary']['total_detections']}
   Detection Quality: {combined['detection_reliability']}
   Risk Level: {combined['overall_risk_level']}

🤖 AI ANALYSIS:
   Status: {'✅ Enhanced with Gemini AI' if combined['ai_enhanced'] else '⚠️ Technical analysis only'}
   
💡 RECOMMENDATIONS:
"""
        
        for rec in combined['recommendations']:
            report += f"   • {rec}\n"
        
        if comprehensive_analysis.get('gemini_analysis', {}).get('status') == 'success':
            report += f"\n🔬 DETAILED AI ANALYSIS:\n{'-'*30}\n"
            report += comprehensive_analysis['gemini_analysis'].get('gemini_analysis', 'No detailed analysis available')
        
        report += f"\n\n{'='*50}\n"
        report += "⚠️  This analysis is for educational purposes. Consult a physician for medical decisions.\n"
        
        return report
    
    def save_analysis(self, comprehensive_analysis: Dict, output_dir: str):
        """Save comprehensive analysis to files."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Generate filename
        image_name = Path(comprehensive_analysis['image_path']).stem
        timestamp = comprehensive_analysis['timestamp'][:19].replace(':', '-')
        base_name = f"ecg_analysis_{image_name}_{timestamp}"
        
        # Save JSON analysis
        json_path = output_path / f"{base_name}.json"
        with open(json_path, 'w') as f:
            json.dump(comprehensive_analysis, f, indent=2, default=str)
        
        # Save summary report
        report_path = output_path / f"{base_name}_summary.txt"
        summary_report = self.generate_summary_report(comprehensive_analysis)
        with open(report_path, 'w') as f:
            f.write(summary_report)
        
        print(f"📄 Analysis saved:")
        print(f"   JSON: {json_path}")
        print(f"   Summary: {report_path}")


def main():
    """Test the ECG Result Analyzer."""
    print("🧪 Testing ECG Result Analyzer...")
    
    # Test without Gemini (technical analysis only)
    analyzer = ECGResultAnalyzer()
    
    # Sample detection results for testing
    sample_detection = {
        'confidences': [0.05, 0.08, 0.12, 0.09, 0.07, 0.11],
        'keypoints': [[0.1, 0.5], [0.25, 0.52], [0.4, 0.48], [0.55, 0.51], [0.7, 0.49], [0.85, 0.53]],
        'all_confidences': [0.01, 0.02, 0.05, 0.08, 0.12, 0.09, 0.07, 0.11, 0.03, 0.02] * 6
    }
    
    # Run technical analysis
    technical_result = analyzer.analyze_detection_results(sample_detection)
    
    print(f"\n📊 Sample Technical Analysis:")
    print(f"   Heart Rate: {technical_result['heart_rate_analysis']['estimated_bpm']} BPM")
    print(f"   Category: {technical_result['heart_rate_analysis']['hr_category']}")
    print(f"   Risk Level: {technical_result['clinical_assessment']['risk_level']}")
    print(f"   Detection Quality: {technical_result['confidence_analysis']['detection_quality']}")
    
    print(f"\n✅ ECG Result Analyzer ready for integration!")


if __name__ == "__main__":
    main()