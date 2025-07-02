#!/usr/bin/env python3
"""
Runner script for GRAPHITE Model Analysis

This script handles dependency installation and runs the comprehensive model analysis.
"""

import subprocess
import sys
import os


def install_dependencies():
    """Install required dependencies for model analysis"""
    print("Installing model analysis dependencies...")
    
    dependencies = [
        "torchinfo",
        "fvcore", 
        "ptflops",
        "timm"
    ]
    
    for dep in dependencies:
        try:
            print(f"Installing {dep}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", dep], 
                                  stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            print(f"✓ {dep} installed successfully")
        except subprocess.CalledProcessError:
            print(f"✗ Failed to install {dep}")


def run_analysis():
    """Run the model analysis"""
    print("\nRunning GRAPHITE model analysis...")
    print("="*60)
    
    try:
        # Import and run the analysis
        from model_analysis import main
        main()
    except ImportError as e:
        print(f"Import error: {e}")
        print("Make sure all dependencies are installed")
    except Exception as e:
        print(f"Analysis error: {e}")


def main():
    """Main runner function"""
    print("GRAPHITE MODEL ANALYSIS RUNNER")
    print("="*60)
    
    # Check if we should install dependencies
    install_deps = input("Install/update dependencies? (y/n): ").lower().strip()
    if install_deps in ['y', 'yes']:
        install_dependencies()
    
    # Run the analysis
    run_analysis()


if __name__ == "__main__":
    main() 