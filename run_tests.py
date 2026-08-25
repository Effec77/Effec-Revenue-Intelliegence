"""Single entry point for running the full test suite: python run_tests.py"""

import sys

import pytest

if __name__ == "__main__":
    exit_code = pytest.main(["tests", "-v"])
    sys.exit(exit_code)
