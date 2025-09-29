#!/usr/bin/env python3
"""
Simple test runner for batch_invariant tests.

This script demonstrates how to run the batch invariant tests
and shows some example usage of the function.
"""

import os
import subprocess
import sys

from vllm.model_executor.layers.batch_invariant import \
    vllm_kernel_override_batch_invariant


def demonstrate_function():
    """Demonstrate the batch invariant function behavior."""
    print("=== Batch Invariant Function Demonstration ===\n")

    test_cases = [
        (None, "Environment variable not set"),
        ("", "Empty string"),
        ("0", "Zero"),
        ("1", "One"),
        ("42", "Positive number"),
        ("-1", "Negative number"),
        ("abc", "Non-numeric string"),
        ("1.5", "Float string"),
        (" 1 ", "Whitespace around number"),
    ]

    for value, description in test_cases:
        # Clean up environment
        if "VLLM_KERNEL_OVERRIDE_BATCH_INVARIANT" in os.environ:
            del os.environ["VLLM_KERNEL_OVERRIDE_BATCH_INVARIANT"]

        # Set environment variable if provided
        if value is not None:
            os.environ["VLLM_KERNEL_OVERRIDE_BATCH_INVARIANT"] = value

        try:
            result = vllm_kernel_override_batch_invariant()
            env_display = f"'{value}'" if value is not None else "unset"
            print(f"VLLM_KERNEL_OVERRIDE_BATCH_INVARIANT={env_display:>8} -> {result} ({description})")
        finally:
            # Clean up
            if "VLLM_KERNEL_OVERRIDE_BATCH_INVARIANT" in os.environ:
                del os.environ["VLLM_KERNEL_OVERRIDE_BATCH_INVARIANT"]


def run_tests():
    """Run the pytest tests."""
    print("\n=== Running Unit Tests ===\n")

    test_file = os.path.join(os.path.dirname(__file__), "test_batch_invariant.py")

    try:
        result = subprocess.run([
            sys.executable, "-m", "pytest", test_file, "-v", "--tb=short"
        ], capture_output=True, text=True)

        print("STDOUT:")
        print(result.stdout)

        if result.stderr:
            print("STDERR:")
            print(result.stderr)

        if result.returncode == 0:
            print("✅ All tests passed!")
        else:
            print("❌ Some tests failed!")

        return result.returncode == 0

    except FileNotFoundError:
        print("❌ pytest not found. Please install pytest to run tests.")
        return False


def main():
    """Main function."""
    print("Batch Invariant Test Runner")
    print("=" * 40)

    # Demonstrate function behavior
    demonstrate_function()

    # Run tests
    success = run_tests()

    # Summary
    print("\n=== Summary ===")
    if success:
        print("✅ All tests completed successfully!")
        print("The batch invariant function is working correctly.")
    else:
        print("❌ Some tests failed. Please check the output above.")

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
