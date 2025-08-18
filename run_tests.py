#!/usr/bin/env python
"""
Test runner for SUM

Run all tests or specific test suites with coverage reporting.
"""

import sys
import pytest
import argparse


def main():
    """Main test runner function."""
    parser = argparse.ArgumentParser(description='Run SUM test suite')
    
    parser.add_argument(
        'tests',
        nargs='*',
        help='Specific test files or directories to run'
    )
    
    parser.add_argument(
        '--slow',
        action='store_true',
        help='Include slow tests'
    )
    
    parser.add_argument(
        '--no-cov',
        action='store_true',
        help='Disable coverage reporting'
    )
    
    parser.add_argument(
        '--html',
        action='store_true',
        help='Generate HTML coverage report'
    )
    
    parser.add_argument(
        '-k',
        '--keyword',
        help='Only run tests matching keyword'
    )
    
    parser.add_argument(
        '-x',
        '--exitfirst',
        action='store_true',
        help='Exit on first failure'
    )
    
    args = parser.parse_args()
    
    # Build pytest arguments
    pytest_args = []
    
    # Add test paths
    if args.tests:
        pytest_args.extend(args.tests)
    else:
        pytest_args.append('tests')
    
    # Add options
    if not args.slow:
        pytest_args.extend(['-m', 'not slow'])
    
    if args.no_cov:
        pytest_args.append('--no-cov')
    
    if args.html:
        pytest_args.append('--cov-report=html')
    
    if args.keyword:
        pytest_args.extend(['-k', args.keyword])
    
    if args.exitfirst:
        pytest_args.append('-x')
    
    # Run tests
    print(f"Running tests with args: {' '.join(pytest_args)}")
    sys.exit(pytest.main(pytest_args))


if __name__ == '__main__':
    main()