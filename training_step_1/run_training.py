#!/usr/bin/env python3
"""
Cross-platform Python launcher for MIL Training
This script provides a cross-platform interface to run the MIL training pipeline
"""

import os
import sys
import argparse
import subprocess
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("MIL_Launcher")

class Colors:
    """ANSI color codes for terminal output"""
    RED = '\033[0;31m'
    GREEN = '\033[0;32m'
    YELLOW = '\033[1;33m'
    BLUE = '\033[0;34m'
    NC = '\033[0m'  # No Color

def print_colored(message, color=Colors.BLUE, prefix="INFO"):
    """Print colored message with prefix"""
    print(f"{color}[{prefix}]{Colors.NC} {message}")

def check_requirements():
    """Check if all requirements are met"""
    print_colored("Checking requirements...", Colors.BLUE)
    
    # Check Python version
    if sys.version_info < (3, 8):
        print_colored(f"Python 3.8+ required. Found: {sys.version}", Colors.RED, "ERROR")
        return False
    
    # Check if we're in the right directory
    script_dir = Path(__file__).parent
    mil_dir = script_dir / "mil_classification"
    
    if not mil_dir.exists():
        print_colored("mil_classification directory not found", Colors.RED, "ERROR")
        return False
    
    # Check if train.py exists
    train_script = mil_dir / "train.py"
    if not train_script.exists():
        print_colored("train.py not found in mil_classification directory", Colors.RED, "ERROR")
        return False
    
    print_colored("Requirements check passed", Colors.GREEN, "SUCCESS")
    return True

def install_dependencies():
    """Install required dependencies"""
    print_colored("Installing dependencies...", Colors.BLUE)
    
    mil_dir = Path(__file__).parent / "mil_classification"
    requirements_file = mil_dir / "requirements.txt"
    
    if requirements_file.exists():
        try:
            subprocess.run([
                sys.executable, "-m", "pip", "install", "-r", str(requirements_file)
            ], check=True, capture_output=True, text=True)
            print_colored("Dependencies installed successfully", Colors.GREEN, "SUCCESS")
            return True
        except subprocess.CalledProcessError as e:
            print_colored(f"Failed to install dependencies: {e}", Colors.RED, "ERROR")
            return False
    else:
        print_colored("requirements.txt not found. Assuming dependencies are installed", Colors.YELLOW, "WARNING")
        return True

def run_quick_test():
    """Run quick training test"""
    print_colored("Running quick training test...", Colors.BLUE)
    
    mil_dir = Path(__file__).parent / "mil_classification"
    quick_test_script = mil_dir / "quick_training_test.py"
    
    if quick_test_script.exists():
        try:
            subprocess.run([
                sys.executable, str(quick_test_script)
            ], check=True, cwd=mil_dir)
            print_colored("Quick test completed successfully", Colors.GREEN, "SUCCESS")
            return True
        except subprocess.CalledProcessError as e:
            print_colored(f"Quick test failed: {e}", Colors.RED, "ERROR")
            return False
    else:
        print_colored("quick_training_test.py not found", Colors.YELLOW, "WARNING")
        return True

def build_training_command(args):
    """Build the training command based on arguments"""
    cmd = [sys.executable, "train.py"]
    
    # Add arguments
    if args.data_dir:
        cmd.extend(["--root_dir", args.data_dir])
    if args.cancer_labels:
        cmd.extend(["--cancer_labels_path", args.cancer_labels])
    if args.normal_labels:
        cmd.extend(["--normal_labels_path", args.normal_labels])
    if args.batch_size:
        cmd.extend(["--batch_size", str(args.batch_size)])
    if args.max_patches:
        cmd.extend(["--max_patches", str(args.max_patches)])
    if args.epochs:
        cmd.extend(["--num_epochs", str(args.epochs)])
    if args.learning_rate:
        cmd.extend(["--learning_rate", str(args.learning_rate)])
    if args.test_size:
        cmd.extend(["--test_size", str(args.test_size)])
    if args.random_state:
        cmd.extend(["--random_state", str(args.random_state)])
    if args.patience:
        cmd.extend(["--early_stopping_patience", str(args.patience)])
    if args.metrics:
        cmd.extend(["--metrics_to_monitor", args.metrics])
    if args.color_norm:
        cmd.append("--use_color_normalization")
    if args.balanced_sampler:
        cmd.append("--use_balanced_sampler")
    
    return cmd

def run_training(cmd, dry_run=False):
    """Run the training command"""
    mil_dir = Path(__file__).parent / "mil_classification"
    
    print_colored("Training command:", Colors.BLUE)
    print(f"  {' '.join(cmd)}")
    print()
    
    if dry_run:
        print_colored("Dry run mode - command not executed", Colors.BLUE)
        return True
    
    # Create output directory
    output_dir = mil_dir / "output"
    output_dir.mkdir(exist_ok=True)
    
    print_colored("Starting training...", Colors.BLUE)
    
    try:
        # Run training
        process = subprocess.run(cmd, cwd=mil_dir, check=True)
        print_colored("Training completed successfully!", Colors.GREEN, "SUCCESS")
        print_colored(f"Check outputs in: {output_dir}", Colors.BLUE)
        return True
    except subprocess.CalledProcessError as e:
        print_colored(f"Training failed with exit code: {e.returncode}", Colors.RED, "ERROR")
        return False
    except KeyboardInterrupt:
        print_colored("Training interrupted by user", Colors.YELLOW, "WARNING")
        return False

def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description="Cross-platform Python launcher for MIL Training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_training.py                                    # Basic training
  python run_training.py --quick_test                      # Quick test
  python run_training.py --epochs 150 --batch_size 16      # Custom parameters
  python run_training.py --dry_run                         # Show command only
        """
    )
    
    # Data parameters
    parser.add_argument('--data_dir', type=str, help='Root directory containing patient folders')
    parser.add_argument('--cancer_labels', type=str, help='Path to cancer patient labels file')
    parser.add_argument('--normal_labels', type=str, help='Path to normal patient labels file')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size for training')
    parser.add_argument('--max_patches', type=int, default=100, help='Maximum patches per patient')
    parser.add_argument('--test_size', type=float, default=0.3, help='Test set proportion')
    parser.add_argument('--random_state', type=int, default=78, help='Random seed')
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--patience', type=int, default=10, help='Early stopping patience')
    parser.add_argument('--metrics', type=str, default='auc', 
                       choices=['accuracy', 'f1', 'auc', 'loss'],
                       help='Metric to monitor for early stopping')
    
    # Options
    parser.add_argument('--color_norm', action='store_true', 
                       help='Enable Macenko color normalization')
    parser.add_argument('--balanced_sampler', action='store_true',
                       help='Use balanced batch sampling')
    parser.add_argument('--quick_test', action='store_true',
                       help='Run quick test (2 epochs, 50 patches)')
    parser.add_argument('--dry_run', action='store_true',
                       help='Show command without executing')
    parser.add_argument('--install_deps', action='store_true',
                       help='Install dependencies and exit')
    
    args = parser.parse_args()
    
    # Quick test overrides
    if args.quick_test:
        args.epochs = 2
        args.max_patches = 50
        args.batch_size = 4
        print_colored("Quick test mode enabled (2 epochs, 50 patches, batch size 4)", Colors.BLUE)
    
    print_colored("Starting MIL Training Pipeline", Colors.BLUE)
    print_colored("=" * 30, Colors.BLUE)
    
    # Check requirements
    if not check_requirements():
        sys.exit(1)
    
    # Install dependencies if requested
    if args.install_deps:
        if install_dependencies():
            print_colored("Dependencies installation completed", Colors.GREEN, "SUCCESS")
            sys.exit(0)
        else:
            sys.exit(1)
    
    # Install dependencies automatically
    if not install_dependencies():
        sys.exit(1)
    
    # Run quick test if requested
    if args.quick_test and not args.dry_run:
        if not run_quick_test():
            print_colored("Quick test failed, proceeding with training anyway...", Colors.YELLOW, "WARNING")
    
    # Build and run training command
    training_cmd = build_training_command(args)
    
    # Display configuration
    print_colored("Training configuration:", Colors.BLUE)
    config_items = [
        ("Data directory", args.data_dir or "Default (PBS_JOBFS)"),
        ("Cancer labels", args.cancer_labels or "Default"),
        ("Normal labels", args.normal_labels or "Default"),
        ("Batch size", args.batch_size),
        ("Max patches", args.max_patches),
        ("Epochs", args.epochs),
        ("Learning rate", args.learning_rate),
        ("Test size", args.test_size),
        ("Random state", args.random_state),
        ("Patience", args.patience),
        ("Metrics", args.metrics),
        ("Color normalization", args.color_norm),
        ("Balanced sampler", args.balanced_sampler),
    ]
    
    for key, value in config_items:
        print(f"  {key}: {value}")
    print()
    
    # Run training
    success = run_training(training_cmd, args.dry_run)
    
    if success:
        print_colored("MIL training pipeline completed successfully!", Colors.GREEN, "SUCCESS")
    else:
        print_colored("MIL training pipeline failed!", Colors.RED, "ERROR")
        sys.exit(1)

if __name__ == "__main__":
    main() 