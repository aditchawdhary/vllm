#!/usr/bin/env python3
"""Simple validation script to check if the test file is syntactically correct."""

import ast
import sys

def validate_test_file():
    """Validate that the test file has correct syntax."""
    try:
        with open('tests/test_ray_lazy_utils.py', 'r') as f:
            content = f.read()

        # Parse the file to check for syntax errors
        ast.parse(content)
        print("✓ Test file syntax is valid")
        return True
    except SyntaxError as e:
        print(f"✗ Syntax error in test file: {e}")
        return False
    except Exception as e:
        print(f"✗ Error validating test file: {e}")
        return False

if __name__ == "__main__":
    success = validate_test_file()
    sys.exit(0 if success else 1)
