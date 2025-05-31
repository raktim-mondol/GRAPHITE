# MIL Model Test Suite

This directory contains the test suite for the Multiple Instance Learning (MIL) model for histopathology image classification.

## Test Structure

The test suite is organized into the following categories:

1. **Model Tests** (`test_model.py`): Unit tests for the MIL model architecture
2. **Data Tests** (`test_data.py`): Unit tests for data loading and processing
3. **Training Tests** (`test_training.py`): Unit tests for the training functions
4. **Integration Tests** (`test_integration.py`): End-to-end tests for the full pipeline

## Running Tests

You can run tests using the `run_tests.bat` script in the root directory:

```
# Run all tests
run_tests

# Run a specific category of tests
run_tests model
run_tests data
run_tests training
run_tests integration

# Run a quick model verification test
run_tests quick

# Run a quick training test (2 epochs)
run_tests quick-train
```

## Test Runner

The test runner is implemented in `run_all_tests.py`. It provides the following options:

```
python -m tests.run_all_tests --help
```

Options:
- `--type`: Type of tests to run (`model`, `data`, `training`, `integration`, `all`)
- `--verbose` or `-v`: Show verbose output

## Quick Training Test

The `quick_training_test.py` script runs a short training session (2 epochs with 50 patches max) to verify that the model can train properly without running a full training cycle.

## Log Files

Test logs are saved to:
- `mil_tests.log`: For unit and integration tests
- `mil_quick_training.log`: For quick training tests 