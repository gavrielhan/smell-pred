#!/usr/bin/env python3
"""
Example script to run ChemBERTa odor classification fine-tuning on HuggingFace Spaces/Colab

This script sets up the environment and runs the fine-tuning with CUDA support.
Perfect for running on HuggingFace Spaces with GPU or Google Colab.
"""

import subprocess
import sys
import os

def install_requirements():
    """Install required packages for HuggingFace environment"""
    print("🔧 Installing required packages...")
    
    packages = [
        "transformers>=4.30.0",
        "torch>=2.0.0",
        "datasets",
        "peft",
        "accelerate",
        "matplotlib",
        "seaborn",
        "scikit-learn",
        "pandas",
        "numpy"
    ]
    
    for package in packages:
        print(f"Installing {package}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
    
    print("✅ All packages installed!")

def setup_data():
    """Download or setup the required data files"""
    print("📊 Setting up data files...")
    
    # Check if data files exist
    required_files = ["goodscents_train.csv", "goodscents_test.csv", "bushdid_predict.csv"]
    missing_files = [f for f in required_files if not os.path.exists(f)]
    
    if missing_files:
        print(f"⚠️  Missing data files: {missing_files}")
        print("Please upload the following files to run the fine-tuning:")
        for file in missing_files:
            print(f"   - {file}")
        return False
    
    print("✅ All data files found!")
    return True

def check_gpu():
    """Check GPU availability"""
    print("🔍 Checking GPU availability...")
    
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            print(f"✅ CUDA GPU detected: {gpu_name}")
            print(f"   GPU Memory: {gpu_memory:.1f} GB")
            return True
        else:
            print("⚠️  No CUDA GPU detected, will use CPU")
            return False
    except ImportError:
        print("❌ PyTorch not installed")
        return False

def run_finetuning():
    """Run the ChemBERTa fine-tuning"""
    print("🚀 Starting ChemBERTa fine-tuning...")
    
    # HuggingFace optimized parameters
    cmd = [
        sys.executable, "chemberta_odor_finetuning.py",
        "--epochs", "10",           # Reduced for faster training
        "--batch_size", "16",       # Good for most GPUs
        "--learning_rate", "5e-4",  # Conservative learning rate
        "--seed", "42"              # Reproducible results
    ]
    
    print(f"Running command: {' '.join(cmd)}")
    
    try:
        subprocess.run(cmd, check=True)
        print("✅ Fine-tuning completed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Fine-tuning failed with error: {e}")
        return False

def main():
    """Main function to orchestrate the fine-tuning on HuggingFace"""
    print("="*80)
    print("🧪 CHEMBERTA ODOR CLASSIFICATION - HUGGINGFACE RUNNER")
    print("="*80)
    
    # Step 1: Install requirements
    try:
        install_requirements()
    except Exception as e:
        print(f"❌ Failed to install requirements: {e}")
        return
    
    # Step 2: Check GPU
    has_gpu = check_gpu()
    
    # Step 3: Setup data
    if not setup_data():
        print("❌ Cannot proceed without data files")
        return
    
    # Step 4: Run fine-tuning
    success = run_finetuning()
    
    if success:
        print("\n" + "="*80)
        print("🎉 SUCCESS! Fine-tuning completed!")
        print("📁 Results saved to: ./chemberta_lora_results/")
        print("📊 Plots available in: ./chemberta_lora_results/plots/")
        print("🤖 Model saved to: ./chemberta_lora_results/final_model/")
        print("="*80)
    else:
        print("\n" + "="*80)
        print("❌ Fine-tuning failed. Check the error messages above.")
        print("="*80)

if __name__ == "__main__":
    main() 