#!/usr/bin/env python3
"""
Gemini Medical Report Generator
Advanced medical report generation for ECG analysis using Gemini AI 2.5 Flash.
"""

import os
import json
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional
import sys

# Add current directory to path for imports
current_dir = Path(__file__).parent.parent
sys.path.append(str(current_dir))

from AIanalysis.ecg_result_analyzer import ECGResultAnalyzer
from AIanalysis.gemini.gemini_analyzer import GeminiECGAnalyzer


class GeminiMedicalReporter:
    """Advanced medical report generator using Gemini AI for ECG analysis."""
    
    def __init__(self, gemini_api_key: Optional[str] = None):
        """Initialize medical reporter with Gemini AI."""
        self.gemini_api_key = gemini_api_key or os.getenv('GEMINI_API_KEY')
        if not self.gemini_api_key:
            raise ValueError("Gemini API key required")
        
        self.analyzer = ECGResultAnalyzer(self.gemini_api_key)
        self.gemini = GeminiECGAnalyzer(self.gemini_api_key)
        
        print("🏥 Gemini Medical Reporter initialized")
        print("📋 Ready for comprehensive ECG medical report generation")
    
    def generate_clinical_narrative(self, image_path: str, detection_results: Dict,
                                  patient_info: Dict = None) -> Dict:
        """Generate comprehensive clinical narrative using Gemini AI."""
        
        # Enhanced prompt for medical narrative
        narrative_prompt = f"""
You are a senior cardiologist writing a comprehensive ECG interpretation report. Based on the ECG image and technical detection data provided, generate a complete clinical narrative that includes:

TECHNICAL DATA PROVIDED:
- R-peaks detected: {len(detection_results.get('confidences', []))}
- Detection confidence: {max(detection_results.get('all_confidences', [0])):.3f}
- Estimated heart rate: ~{len(detection_results.get('confidences', [])) * 6} BPM

PATIENT INFORMATION:
{json.dumps(patient_info, indent=2) if patient_info else "No patient information provided"}

REQUIRED REPORT SECTIONS:

1. CLINICAL IMPRESSION:
   Write a 2-3 sentence clinical impression summarizing the overall ECG findings.

2. DETAILED INTERPRETATION:
   - RHYTHM: Describe the cardiac rhythm (sinus, atrial fibrillation, etc.)
   - RATE: Comment on heart rate and any variations
   - AXIS: Evaluate cardiac axis if determinable
   - INTERVALS: Comment on PR, QRS, QT intervals
   - MORPHOLOGY: Describe P-waves, QRS complexes, T-waves
   - ST-SEGMENTS: Evaluate for elevation, depression, or changes

3. ABNORMAL FINDINGS:
   List any abnormal findings with clinical significance. If no abnormalities, state "No acute abnormalities identified."

4. CLINICAL CORRELATION:
   Suggest clinical correlation or additional testing if indicated.

5. RISK STRATIFICATION:
   - IMMEDIATE RISK: LOW / MODERATE / HIGH
   - URGENT ACTIONS: Any immediate interventions needed
   - FOLLOW-UP: Recommended follow-up care

6. DIFFERENTIAL DIAGNOSES:
   List possible diagnoses based on ECG findings (if abnormal findings present).

7. RECOMMENDATIONS:
   - Immediate management recommendations
   - Follow-up testing suggestions
   - Patient counseling points

Please write in professional medical terminology suitable for a cardiologist's report, but ensure clarity for clinical decision-making.
"""
        
        # Generate analysis with enhanced prompt
        narrative_result = self.gemini.analyze_ecg(
            image_path, detection_results, narrative_prompt
        )
        
        return narrative_result
    
    def create_structured_report(self, comprehensive_analysis: Dict,
                               patient_info: Dict = None) -> Dict:
        """Create a structured medical report from comprehensive analysis."""
        
        report_data = {
            'report_header': {
                'report_type': 'ECG Analysis Report',
                'generated_by': 'Gemini AI 2.5 Flash Medical System',
                'generation_date': datetime.now().isoformat(),
                'report_id': f"ECG_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            },
            'patient_information': patient_info or {
                'name': 'Patient Name Not Provided',
                'age': 'Unknown',
                'gender': 'Unknown',
                'medical_record_number': 'N/A'
            },
            'technical_summary': comprehensive_analysis.get('technical_analysis', {}),
            'ai_interpretation': comprehensive_analysis.get('gemini_analysis', {}),
            'clinical_assessment': self._extract_clinical_findings(comprehensive_analysis),
            'recommendations': self._extract_recommendations(comprehensive_analysis),
            'quality_metrics': {
                'detection_reliability': comprehensive_analysis.get('combined_assessment', {}).get('detection_reliability', 'Unknown'),
                'ai_confidence': 'High' if comprehensive_analysis.get('gemini_analysis', {}).get('status') == 'success' else 'Low',
                'technical_quality': self._assess_technical_quality(comprehensive_analysis)
            }
        }
        
        return report_data
    
    def _extract_clinical_findings(self, analysis: Dict) -> Dict:
        """Extract clinical findings from comprehensive analysis."""
        
        tech_analysis = analysis.get('technical_analysis', {})
        combined = analysis.get('combined_assessment', {})
        
        findings = {
            'heart_rate': {
                'value': tech_analysis.get('heart_rate_analysis', {}).get('estimated_bpm', 'Unknown'),
                'category': tech_analysis.get('heart_rate_analysis', {}).get('hr_category', 'Unknown'),
                'assessment': tech_analysis.get('heart_rate_analysis', {}).get('hr_assessment', 'No assessment available')
            },
            'rhythm_analysis': tech_analysis.get('rhythm_analysis', {}),
            'risk_level': combined.get('overall_risk_level', 'Unknown'),
            'urgent_findings': self._identify_urgent_findings(tech_analysis),
            'clinical_significance': self._determine_clinical_significance(analysis)
        }
        
        return findings
    
    def _extract_recommendations(self, analysis: Dict) -> List[Dict]:
        """Extract and structure recommendations."""
        
        base_recommendations = analysis.get('combined_assessment', {}).get('recommendations', [])
        tech_analysis = analysis.get('technical_analysis', {})
        
        structured_recommendations = []
        
        for rec in base_recommendations:
            priority = 'ROUTINE'
            if 'immediate' in rec.lower() or 'urgent' in rec.lower():
                priority = 'URGENT'
            elif 'follow-up' in rec.lower() or 'monitor' in rec.lower():
                priority = 'MODERATE'
            
            structured_recommendations.append({
                'recommendation': rec,
                'priority': priority,
                'category': self._categorize_recommendation(rec)
            })
        
        # Add Gemini-specific recommendations if available
        gemini_analysis = analysis.get('gemini_analysis', {})
        if gemini_analysis.get('status') == 'success':
            structured_recommendations.append({
                'recommendation': 'Comprehensive AI analysis completed - refer to detailed interpretation',
                'priority': 'ROUTINE',
                'category': 'AI_ANALYSIS'
            })
        
        return structured_recommendations
    
    def _categorize_recommendation(self, recommendation: str) -> str:
        """Categorize recommendation type."""
        rec_lower = recommendation.lower()
        
        if 'medical' in rec_lower or 'physician' in rec_lower:
            return 'MEDICAL_CONSULTATION'
        elif 'monitor' in rec_lower or 'follow-up' in rec_lower:
            return 'MONITORING'
        elif 'test' in rec_lower or 'ecg' in rec_lower:
            return 'DIAGNOSTIC_TESTING'
        elif 'review' in rec_lower:
            return 'MANUAL_REVIEW'
        else:
            return 'GENERAL'
    
    def _identify_urgent_findings(self, tech_analysis: Dict) -> List[str]:
        """Identify findings requiring urgent attention."""
        
        urgent_findings = []
        
        # Check heart rate
        hr_bpm = tech_analysis.get('heart_rate_analysis', {}).get('estimated_bpm', 0)
        if hr_bpm < 50:
            urgent_findings.append("Severe bradycardia detected")
        elif hr_bpm > 150:
            urgent_findings.append("Severe tachycardia detected")
        
        # Check risk level
        risk_level = tech_analysis.get('clinical_assessment', {}).get('risk_level', 'LOW')
        if risk_level == 'HIGH':
            urgent_findings.append("High-risk ECG pattern identified")
        
        # Check detection quality
        confidence = tech_analysis.get('detection_summary', {}).get('mean_confidence', 1.0)
        if confidence < 0.1:
            urgent_findings.append("Poor signal quality - manual interpretation required")
        
        return urgent_findings
    
    def _determine_clinical_significance(self, analysis: Dict) -> str:
        """Determine overall clinical significance."""
        
        risk_level = analysis.get('combined_assessment', {}).get('overall_risk_level', 'LOW')
        urgent_findings = self._identify_urgent_findings(analysis.get('technical_analysis', {}))
        
        if risk_level == 'HIGH' or len(urgent_findings) > 0:
            return "HIGH - Immediate clinical attention recommended"
        elif risk_level == 'MODERATE':
            return "MODERATE - Follow-up care indicated"
        else:
            return "LOW - Routine monitoring sufficient"
    
    def _assess_technical_quality(self, analysis: Dict) -> str:
        """Assess technical quality of analysis."""
        
        gemini_success = analysis.get('gemini_analysis', {}).get('status') == 'success'
        tech_confidence = analysis.get('technical_analysis', {}).get('detection_summary', {}).get('mean_confidence', 0)
        
        if gemini_success and tech_confidence > 0.3:
            return "HIGH - AI and technical analysis reliable"
        elif gemini_success or tech_confidence > 0.2:
            return "MODERATE - Partially reliable analysis"
        else:
            return "LOW - Manual review strongly recommended"
    
    def generate_professional_report(self, image_path: str, detection_results: Dict,
                                   patient_info: Dict = None, output_path: str = None) -> str:
        """Generate a complete professional medical report."""
        
        print(f"📋 Generating professional medical report for {Path(image_path).name}")
        
        # Run comprehensive analysis
        comprehensive_analysis = self.analyzer.analyze_with_gemini(
            image_path, detection_results, patient_info
        )
        
        # Create structured report
        structured_report = self.create_structured_report(comprehensive_analysis, patient_info)
        
        # Generate narrative report
        narrative = self.generate_clinical_narrative(image_path, detection_results, patient_info)
        
        # Format professional report
        report = self._format_professional_report(structured_report, narrative, comprehensive_analysis)
        
        # Save report if output path provided
        if output_path:
            output_dir = Path(output_path).parent
            output_dir.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, 'w') as f:
                f.write(report)
            
            # Save JSON data
            json_path = str(output_path).replace('.txt', '_data.json')
            with open(json_path, 'w') as f:
                json.dump({
                    'structured_report': structured_report,
                    'comprehensive_analysis': comprehensive_analysis
                }, f, indent=2, default=str)
            
            print(f"📄 Professional report saved: {output_path}")
            print(f"📊 Data saved: {json_path}")
        
        return report
    
    def _format_professional_report(self, structured: Dict, narrative: Dict, 
                                  comprehensive: Dict) -> str:
        """Format professional medical report."""
        
        header = structured['report_header']
        patient = structured['patient_information']
        clinical = structured['clinical_assessment']
        
        report = f"""
╔════════════════════════════════════════════════════════════════════╗
║                     ELECTROCARDIOGRAM ANALYSIS REPORT              ║
║                        Gemini AI Medical System                     ║
╚════════════════════════════════════════════════════════════════════╝

REPORT INFORMATION
──────────────────────────────────────────────────────────────────────
Report ID: {header['report_id']}
Generated: {datetime.fromisoformat(header['generation_date']).strftime('%B %d, %Y at %H:%M:%S')}
System: {header['generated_by']}

PATIENT INFORMATION  
──────────────────────────────────────────────────────────────────────
Name: {patient['name']}
Age: {patient['age']}
Gender: {patient['gender']}
MRN: {patient['medical_record_number']}

TECHNICAL SUMMARY
──────────────────────────────────────────────────────────────────────
Heart Rate: {clinical['heart_rate']['value']} BPM ({clinical['heart_rate']['category']})
Risk Level: {clinical['risk_level']}
Clinical Significance: {clinical['clinical_significance']}
Analysis Quality: {structured['quality_metrics']['technical_quality']}

"""
        
        # Add urgent findings if present
        if clinical['urgent_findings']:
            report += "⚠️  URGENT FINDINGS:\n"
            for finding in clinical['urgent_findings']:
                report += f"   • {finding}\n"
            report += "\n"
        
        # Add Gemini AI interpretation if available
        if narrative.get('status') == 'success':
            report += "GEMINI AI CLINICAL INTERPRETATION\n"
            report += "──────────────────────────────────────────────────────────────────────\n"
            report += f"{narrative.get('gemini_analysis', 'No detailed analysis available')}\n\n"
        
        # Add recommendations
        report += "CLINICAL RECOMMENDATIONS\n"
        report += "──────────────────────────────────────────────────────────────────────\n"
        
        for rec in structured['recommendations']:
            priority_symbol = "🔴" if rec['priority'] == 'URGENT' else "🟡" if rec['priority'] == 'MODERATE' else "🟢"
            report += f"{priority_symbol} {rec['recommendation']}\n"
        
        report += "\n"
        
        # Footer
        report += "══════════════════════════════════════════════════════════════════════\n"
        report += "IMPORTANT MEDICAL DISCLAIMER:\n"
        report += "This report is generated by AI for educational and assistive purposes.\n"
        report += "All clinical decisions should be made by qualified medical professionals.\n"
        report += "This analysis does not replace clinical judgment or consultation.\n"
        report += "══════════════════════════════════════════════════════════════════════\n"
        
        return report


def main():
    """Command-line interface for medical report generation."""
    parser = argparse.ArgumentParser(description='Generate professional ECG medical reports with Gemini AI')
    parser.add_argument('--image', required=True, help='ECG image path')
    parser.add_argument('--detection-results', required=True, help='JSON file with detection results')
    parser.add_argument('--patient-info', help='JSON file with patient information')
    parser.add_argument('--output', help='Output report path')
    parser.add_argument('--gemini-key', help='Gemini API key (or set GEMINI_API_KEY env variable)')
    
    args = parser.parse_args()
    
    # Load detection results
    try:
        with open(args.detection_results, 'r') as f:
            detection_results = json.load(f)
    except Exception as e:
        print(f"❌ Failed to load detection results: {e}")
        return
    
    # Load patient info if provided
    patient_info = None
    if args.patient_info:
        try:
            with open(args.patient_info, 'r') as f:
                patient_info = json.load(f)
        except Exception as e:
            print(f"⚠️  Failed to load patient info: {e}")
    
    # Initialize reporter
    try:
        reporter = GeminiMedicalReporter(args.gemini_key)
    except ValueError as e:
        print(f"❌ {e}")
        return
    
    # Generate report
    output_path = args.output or f"medical_report_{Path(args.image).stem}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    
    report = reporter.generate_professional_report(
        args.image, detection_results, patient_info, output_path
    )
    
    print(f"\n📋 Professional medical report generated!")
    print(f"🏥 Report ready for clinical review")


if __name__ == "__main__":
    main()