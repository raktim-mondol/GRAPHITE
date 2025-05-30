@echo off
REM MIL Training Script for Windows - Histopathology Image Classification
REM This script provides an easy interface to run the MIL training pipeline on Windows

setlocal enabledelayedexpansion

REM Set script directory
set "SCRIPT_DIR=%~dp0"
set "MIL_DIR=%SCRIPT_DIR%mil_classification"

REM Color codes for output (Windows 10+)
set "RED=[91m"
set "GREEN=[92m"
set "YELLOW=[93m"
set "BLUE=[94m"
set "NC=[0m"

REM Default parameters
set "DEFAULT_BATCH_SIZE=8"
set "DEFAULT_MAX_PATCHES=100"
set "DEFAULT_EPOCHS=100"
set "DEFAULT_LEARNING_RATE=0.001"
set "DEFAULT_TEST_SIZE=0.3"
set "DEFAULT_RANDOM_STATE=78"
set "DEFAULT_PATIENCE=10"
set "DEFAULT_METRICS=auc"

echo %BLUE%[INFO]%NC% Starting MIL Training Pipeline
echo %BLUE%[INFO]%NC% ==============================

REM Check if mil_classification directory exists
if not exist "%MIL_DIR%" (
    echo %RED%[ERROR]%NC% mil_classification directory not found. Please run this script from training_step_1 directory.
    pause
    exit /b 1
)

REM Check if train.py exists
if not exist "%MIL_DIR%\train.py" (
    echo %RED%[ERROR]%NC% train.py not found in mil_classification directory.
    pause
    exit /b 1
)

echo %GREEN%[SUCCESS]%NC% Requirements check passed.

REM Change to mil_classification directory
cd /d "%MIL_DIR%"

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo %RED%[ERROR]%NC% Python not found. Please install Python 3.8+ and add it to PATH.
    pause
    exit /b 1
)

echo %BLUE%[INFO]%NC% Installing dependencies...
if exist "requirements.txt" (
    pip install -r requirements.txt
    if errorlevel 1 (
        echo %RED%[ERROR]%NC% Failed to install dependencies.
        pause
        exit /b 1
    )
    echo %GREEN%[SUCCESS]%NC% Dependencies installed successfully.
) else (
    echo %YELLOW%[WARNING]%NC% requirements.txt not found. Assuming dependencies are already installed.
)

REM Create output directory
if not exist "output" mkdir output

REM Check for command line arguments
set "QUICK_TEST=false"
set "USE_COLOR_NORM="
set "USE_BALANCED_SAMPLER="

if "%1"=="--quick_test" (
    set "QUICK_TEST=true"
    echo %BLUE%[INFO]%NC% Quick test mode enabled
)

if "%1"=="--help" (
    echo.
    echo Usage: run_training.bat [OPTIONS]
    echo.
    echo OPTIONS:
    echo   --quick_test           Run quick test (2 epochs, 50 patches)
    echo   --color_norm          Enable Macenko color normalization
    echo   --balanced_sampler    Use balanced batch sampling
    echo   --help                Show this help message
    echo.
    echo Examples:
    echo   run_training.bat                    # Basic training
    echo   run_training.bat --quick_test       # Quick test run
    echo.
    pause
    exit /b 0
)

if "%1"=="--color_norm" set "USE_COLOR_NORM=--use_color_normalization"
if "%2"=="--color_norm" set "USE_COLOR_NORM=--use_color_normalization"

if "%1"=="--balanced_sampler" set "USE_BALANCED_SAMPLER=--use_balanced_sampler"
if "%2"=="--balanced_sampler" set "USE_BALANCED_SAMPLER=--use_balanced_sampler"

REM Build training command
set "TRAIN_CMD=python train.py"

if "%QUICK_TEST%"=="true" (
    set "TRAIN_CMD=%TRAIN_CMD% --num_epochs 2 --max_patches 50 --batch_size 4"
    echo %BLUE%[INFO]%NC% Running quick training test (2 epochs, 50 patches, batch size 4)
) else (
    set "TRAIN_CMD=%TRAIN_CMD% --num_epochs %DEFAULT_EPOCHS% --max_patches %DEFAULT_MAX_PATCHES% --batch_size %DEFAULT_BATCH_SIZE%"
)

set "TRAIN_CMD=%TRAIN_CMD% --learning_rate %DEFAULT_LEARNING_RATE%"
set "TRAIN_CMD=%TRAIN_CMD% --test_size %DEFAULT_TEST_SIZE%"
set "TRAIN_CMD=%TRAIN_CMD% --random_state %DEFAULT_RANDOM_STATE%"
set "TRAIN_CMD=%TRAIN_CMD% --early_stopping_patience %DEFAULT_PATIENCE%"
set "TRAIN_CMD=%TRAIN_CMD% --metrics_to_monitor %DEFAULT_METRICS%"

if defined USE_COLOR_NORM set "TRAIN_CMD=%TRAIN_CMD% %USE_COLOR_NORM%"
if defined USE_BALANCED_SAMPLER set "TRAIN_CMD=%TRAIN_CMD% %USE_BALANCED_SAMPLER%"

echo %BLUE%[INFO]%NC% Training configuration:
echo   Epochs: %DEFAULT_EPOCHS%
echo   Batch size: %DEFAULT_BATCH_SIZE%
echo   Max patches: %DEFAULT_MAX_PATCHES%
echo   Learning rate: %DEFAULT_LEARNING_RATE%
echo   Test size: %DEFAULT_TEST_SIZE%
echo   Random state: %DEFAULT_RANDOM_STATE%
echo   Patience: %DEFAULT_PATIENCE%
echo   Metrics to monitor: %DEFAULT_METRICS%
echo   Color normalization: %USE_COLOR_NORM%
echo   Balanced sampler: %USE_BALANCED_SAMPLER%
echo.

echo %BLUE%[INFO]%NC% Command to execute:
echo   %TRAIN_CMD%
echo.

echo %BLUE%[INFO]%NC% Starting training...
%TRAIN_CMD%

if errorlevel 1 (
    echo %RED%[ERROR]%NC% Training failed.
    pause
    exit /b 1
)

echo %GREEN%[SUCCESS]%NC% Training completed successfully!
echo %BLUE%[INFO]%NC% Check outputs in: %MIL_DIR%\output\
echo.

pause 