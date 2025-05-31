"""
Unified test runner for MIL model.
This script runs all tests or specific test categories based on command line arguments.
"""

import unittest
import argparse
import sys
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("mil_tests.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("MIL_Tests")

def run_tests(test_type=None, verbose=False):
    """
    Run tests based on the specified type.
    
    Args:
        test_type (str, optional): Type of tests to run ('model', 'data', 'training', 'integration', or None for all)
        verbose (bool): Whether to show verbose output
    
    Returns:
        bool: True if all tests pass, False otherwise
    """
    # Add the parent directory to sys.path to ensure imports work correctly
    parent_dir = str(Path(__file__).parent.parent)
    if parent_dir not in sys.path:
        sys.path.insert(0, parent_dir)
    
    # Determine which tests to run
    if test_type == 'model':
        from tests import test_model
        test_suite = unittest.defaultTestLoader.loadTestsFromModule(test_model)
    elif test_type == 'data':
        from tests import test_data
        test_suite = unittest.defaultTestLoader.loadTestsFromModule(test_data)
    elif test_type == 'training':
        from tests import test_training
        test_suite = unittest.defaultTestLoader.loadTestsFromModule(test_training)
    elif test_type == 'integration':
        from tests import test_integration
        test_suite = unittest.defaultTestLoader.loadTestsFromModule(test_integration)
    else:
        # Run all tests
        test_suite = unittest.defaultTestLoader.discover('tests')
    
    # Run the tests
    verbosity = 2 if verbose else 1
    test_runner = unittest.TextTestRunner(verbosity=verbosity)
    result = test_runner.run(test_suite)
    
    return len(result.failures) == 0 and len(result.errors) == 0

def main():
    parser = argparse.ArgumentParser(description='Run MIL model tests')
    parser.add_argument('--type', type=str, choices=['model', 'data', 'training', 'integration', 'all'],
                        default='all', help='Type of tests to run')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Show verbose output')
    args = parser.parse_args()
    
    test_type = None if args.type == 'all' else args.type
    
    logger.info(f"Running {'all' if test_type is None else test_type} tests...")
    
    success = run_tests(test_type, args.verbose)
    
    if success:
        logger.info("All tests passed!")
        sys.exit(0)
    else:
        logger.error("Some tests failed!")
        sys.exit(1)

if __name__ == "__main__":
    main() 