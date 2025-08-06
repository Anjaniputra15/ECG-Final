#!/usr/bin/env python3
"""
Gemini AI Setup and Configuration
Easy setup script for Gemini AI integration with ECG analysis system.
"""

import os
import sys
from pathlib import Path


def setup_gemini_integration():
    """Setup Gemini AI integration for ECG analysis."""
    
    print("🤖 Setting up Gemini AI Integration for ECG Analysis")
    print("=" * 60)
    
    # Check if API key is already set
    existing_key = os.getenv('GEMINI_API_KEY')
    if existing_key:
        print(f"✅ Gemini API key already configured: {'*' * 10}{existing_key[-4:]}")
    else:
        print("⚠️  No Gemini API key found in environment")
        print("\nTo get your Gemini API key:")
        print("1. Visit: https://ai.google.dev/")
        print("2. Click 'Get API Key in Google AI Studio'")
        print("3. Create a new API key")
        print("4. Copy the key and set it as environment variable:")
        print("   export GEMINI_API_KEY='your-api-key-here'")
        print("   (Add this to your ~/.bashrc or ~/.zshrc for persistence)")
    
    print("\n🏗️  Gemini Integration Files Created:")
    print(f"   📁 AIanalysis/")
    print(f"   ├── gemini/")
    print(f"   │   └── gemini_analyzer.py       - Core Gemini AI interface")
    print(f"   ├── ecg_result_analyzer.py       - ECG analysis bridge")
    print(f"   └── gemini_medical_reporter.py   - Medical report generator")
    
    print("\n🧪 Usage Examples:")
    print("\n1. Test with single ECG image + Gemini AI:")
    print("   python test_live_model.py --model experiments/bootstrap_training_*/final_trained_model.pth \\")
    print("                              --image your_ecg.png \\")  
    print("                              --gemini")
    
    print("\n2. Test Gemini connection:")
    print("   python AIanalysis/gemini/gemini_analyzer.py")
    
    print("\n3. Generate professional medical report:")
    print("   python AIanalysis/gemini_medical_reporter.py \\")
    print("          --image ecg.png \\")
    print("          --detection-results results.json \\")
    print("          --output medical_report.txt")
    
    print("\n📋 Expected Workflow:")
    print("   1. Run R-peak detection: Your trained model detects R-peaks")
    print("   2. Gemini analysis: AI analyzes ECG image + detection data")
    print("   3. Medical report: Comprehensive clinical interpretation")
    print("   4. Clinical decision: Healthcare professional review")
    
    print("\n🔑 Environment Setup:")
    if not existing_key:
        api_key = input("\nEnter your Gemini API key (or press Enter to skip): ").strip()
        if api_key:
            # Set for current session
            os.environ['GEMINI_API_KEY'] = api_key
            print(f"✅ API key set for current session")
            print("💡 To make permanent, add to your shell profile:")
            print(f"   echo 'export GEMINI_API_KEY=\"{api_key}\"' >> ~/.bashrc")
        else:
            print("⚠️  Skipped API key setup - set GEMINI_API_KEY environment variable later")
    
    # Test Gemini connection if API key available
    test_key = api_key if 'api_key' in locals() and api_key else existing_key
    if test_key:
        print("\n🧪 Testing Gemini Connection...")
        try:
            from AIanalysis.gemini.gemini_analyzer import test_gemini_connection
            if test_gemini_connection(test_key):
                print("🎉 Gemini AI Integration Ready!")
                print("\nNext Steps:")
                print("1. Test your trained model: python test_live_model.py --help")
                print("2. Run with Gemini: python test_live_model.py ... --gemini")
                print("3. Generate reports: python AIanalysis/gemini_medical_reporter.py --help")
            else:
                print("❌ Gemini connection failed - check your API key")
        except Exception as e:
            print(f"⚠️  Gemini test failed: {e}")
    
    print("\n" + "=" * 60)
    print("🏥 Gemini AI ECG Analysis System Setup Complete!")


if __name__ == "__main__":
    setup_gemini_integration()