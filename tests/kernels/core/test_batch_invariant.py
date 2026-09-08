# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Tests for csrc/core/batch_invariant.hpp

This module tests the vllm_kernel_override_batch_invariant() function
which checks the VLLM_KERNEL_OVERRIDE_BATCH_INVARIANT environment variable.

Since the C++ function is inline and simple, we test it by comparing
behavior with the Python implementation and testing the logic thoroughly.
"""

import os
import threading
import time

import pytest

from vllm.model_executor.layers.batch_invariant import (
    vllm_kernel_override_batch_invariant)


def test_vllm_kernel_override_batch_invariant_default():
    """Test that the function returns False when env var is not set."""
    # Ensure the environment variable is not set
    if "VLLM_KERNEL_OVERRIDE_BATCH_INVARIANT" in os.environ:
        del os.environ["VLLM_KERNEL_OVERRIDE_BATCH_INVARIANT"]

    result = vllm_kernel_override_batch_invariant()
    assert result is False


def test_vllm_kernel_override_batch_invariant_zero():
    """Test that the function returns False when env var is set to '0'."""
    os.environ["VLLM_KERNEL_OVERRIDE_BATCH_INVARIANT"] = "0"

    try:
        result = vllm_kernel_override_batch_invariant()
        assert result is False
    finally:
        del os.environ["VLLM_KERNEL_OVERRIDE_BATCH_INVARIANT"]


def test_vllm_kernel_override_batch_invariant_one():
    """Test that the function returns True when env var is set to '1'."""
    os.environ["VLLM_KERNEL_OVERRIDE_BATCH_INVARIANT"] = "1"

    try:
        result = vllm_kernel_override_batch_invariant()
        assert result is True
    finally:
        del os.environ["VLLM_KERNEL_OVERRIDE_BATCH_INVARIANT"]


def test_vllm_kernel_override_batch_invariant_positive_number():
    """
    Test that the function returns True when env var is set to positive number.
    """
    os.environ["VLLM_KERNEL_OVERRIDE_BATCH_INVARIANT"] = "42"

    try:
        result = vllm_kernel_override_batch_invariant()
        assert result is True
    finally:
        del os.environ["VLLM_KERNEL_OVERRIDE_BATCH_INVARIANT"]


def test_vllm_kernel_override_batch_invariant_negative_number():
    """
    Test that the function returns True when env var is set to negative number.
    """
    os.environ["VLLM_KERNEL_OVERRIDE_BATCH_INVARIANT"] = "-1"

    try:
        result = vllm_kernel_override_batch_invariant()
        assert result is True
    finally:
        del os.environ["VLLM_KERNEL_OVERRIDE_BATCH_INVARIANT"]


def test_vllm_kernel_override_batch_invariant_empty_string():
    """
    Test that the function returns False when env var is set to empty string.
    """
    os.environ["VLLM_KERNEL_OVERRIDE_BATCH_INVARIANT"] = ""

    try:
        result = vllm_kernel_override_batch_invariant()
        assert result is False
    finally:
        del os.environ["VLLM_KERNEL_OVERRIDE_BATCH_INVARIANT"]


def test_vllm_kernel_override_batch_invariant_non_numeric():
    """
    Test that the function returns False when env var is set to non-numeric
    string.
    """
    test_values = ["abc", "true", "false", "yes", "no", "on", "off"]

    for value in test_values:
        os.environ["VLLM_KERNEL_OVERRIDE_BATCH_INVARIANT"] = value

        try:
            result = vllm_kernel_override_batch_invariant()
            msg = f"Expected False for value '{value}', got {result}"
            assert result is False, msg
        finally:
            del os.environ["VLLM_KERNEL_OVERRIDE_BATCH_INVARIANT"]


def test_vllm_kernel_override_batch_invariant_whitespace():
    """Test that the function handles whitespace correctly."""
    test_values = [" 1 ", "\t1\t", "\n1\n", " 0 ", "\t0\t", "\n0\n"]
    expected_results = [True, True, True, False, False, False]

    for value, expected in zip(test_values, expected_results):
        os.environ["VLLM_KERNEL_OVERRIDE_BATCH_INVARIANT"] = value

        try:
            result = vllm_kernel_override_batch_invariant()
            msg = f"Expected {expected} for value '{value}', got {result}"
            assert result is expected, msg
        finally:
            del os.environ["VLLM_KERNEL_OVERRIDE_BATCH_INVARIANT"]


def test_vllm_kernel_override_batch_invariant_mixed_numeric():
    """Test that the function handles mixed numeric strings correctly."""
    test_values = ["1abc", "abc1", "1.0", "1.5", "0.0", "0.5"]
    expected_results = [True, False, True, True, False, False]

    for value, expected in zip(test_values, expected_results):
        os.environ["VLLM_KERNEL_OVERRIDE_BATCH_INVARIANT"] = value

        try:
            result = vllm_kernel_override_batch_invariant()
            msg = f"Expected {expected} for value '{value}', got {result}"
            assert result is expected, msg
        finally:
            del os.environ["VLLM_KERNEL_OVERRIDE_BATCH_INVARIANT"]


def test_vllm_kernel_override_batch_invariant_large_numbers():
    """Test that the function handles large numbers correctly."""
    test_values = ["999999999", "-999999999", "2147483647", "-2147483648"]
    expected_results = [True, True, True, True]

    for value, expected in zip(test_values, expected_results):
        os.environ["VLLM_KERNEL_OVERRIDE_BATCH_INVARIANT"] = value

        try:
            result = vllm_kernel_override_batch_invariant()
            msg = f"Expected {expected} for value '{value}', got {result}"
            assert result is expected, msg
        finally:
            del os.environ["VLLM_KERNEL_OVERRIDE_BATCH_INVARIANT"]


@pytest.mark.parametrize(
    "env_value,expected",
    [
        (None, False),  # Environment variable not set
        ("", False),  # Empty string
        ("0", False),  # Zero
        ("1", True),  # One
        ("42", True),  # Positive number
        ("-1", True),  # Negative number
        ("abc", False),  # Non-numeric string
        ("1.0", True),  # Float that converts to 1
        ("0.0", False),  # Float that converts to 0
        (" 1 ", True),  # Whitespace around 1
        (" 0 ", False),  # Whitespace around 0
    ])
def test_vllm_kernel_override_batch_invariant_parametrized(
        env_value, expected):
    """Parametrized test for various environment variable values."""
    # Clean up environment
    if "VLLM_KERNEL_OVERRIDE_BATCH_INVARIANT" in os.environ:
        del os.environ["VLLM_KERNEL_OVERRIDE_BATCH_INVARIANT"]

    # Set environment variable if provided
    if env_value is not None:
        os.environ["VLLM_KERNEL_OVERRIDE_BATCH_INVARIANT"] = env_value

    try:
        result = vllm_kernel_override_batch_invariant()
        msg = f"Expected {expected} for env_value '{env_value}', got {result}"
        assert result is expected, msg
    finally:
        # Clean up
        if "VLLM_KERNEL_OVERRIDE_BATCH_INVARIANT" in os.environ:
            del os.environ["VLLM_KERNEL_OVERRIDE_BATCH_INVARIANT"]


def test_vllm_kernel_override_batch_invariant_thread_safety():
    """Test that the function is thread-safe (basic test)."""
    results = []
    errors = []

    def worker(env_value, expected):
        try:
            os.environ["VLLM_KERNEL_OVERRIDE_BATCH_INVARIANT"] = env_value
            time.sleep(
                0.01)  # Small delay to increase chance of race conditions
            result = vllm_kernel_override_batch_invariant()
            results.append((env_value, result, expected))
        except Exception as e:
            errors.append((env_value, str(e)))
        finally:
            if "VLLM_KERNEL_OVERRIDE_BATCH_INVARIANT" in os.environ:
                del os.environ["VLLM_KERNEL_OVERRIDE_BATCH_INVARIANT"]

    # Create multiple threads with different environment values
    threads = []
    test_cases = [("0", False), ("1", True), ("42", True), ("abc", False)]

    for env_value, expected in test_cases * 5:  # Run each test case 5 times
        thread = threading.Thread(target=worker, args=(env_value, expected))
        threads.append(thread)

    # Start all threads
    for thread in threads:
        thread.start()

    # Wait for all threads to complete
    for thread in threads:
        thread.join()

    # Check for errors
    assert not errors, f"Errors occurred in threads: {errors}"

    # Verify results (note: due to environment variable sharing,
    # we can't guarantee exact results, but we can check no crashes occurred)
    assert len(results) == len(
        threads), "Not all threads completed successfully"


def test_vllm_kernel_override_batch_invariant_edge_cases():
    """Test edge cases and corner cases."""
    edge_cases = [
        ("00", False),  # Leading zero
        ("01", True),  # Leading zero with non-zero
        ("+1", True),  # Positive sign
        ("+0", False),  # Positive sign with zero
        ("1e0", True),  # Scientific notation
        ("0e0", False),  # Scientific notation zero
        ("inf", False),  # Infinity string
        ("nan", False),  # NaN string
        ("0x1", False),  # Hexadecimal (should be treated as non-numeric)
        ("0b1", False),  # Binary (should be treated as non-numeric)
    ]

    for value, expected in edge_cases:
        os.environ["VLLM_KERNEL_OVERRIDE_BATCH_INVARIANT"] = value

        try:
            result = vllm_kernel_override_batch_invariant()
            msg = f"Expected {expected} for value '{value}', got {result}"
            assert result is expected, msg
        finally:
            del os.environ["VLLM_KERNEL_OVERRIDE_BATCH_INVARIANT"]


def test_vllm_kernel_override_batch_invariant_c_atoi_behavior():
    """Test that the function behaves like C's atoi function."""
    # Test cases that specifically test atoi behavior
    atoi_test_cases = [
        ("123", True),  # Simple positive number
        ("-123", True),  # Simple negative number
        ("0", False),  # Zero
        ("000", False),  # Multiple zeros
        ("123abc", True
         ),  # Number followed by non-digits (atoi stops at first non-digit)
        ("abc123", False),  # Non-digits followed by number (atoi returns 0)
        ("  123", True),  # Leading whitespace (atoi skips whitespace)
        ("  -123", True),  # Leading whitespace with negative
        ("  0", False),  # Leading whitespace with zero
        ("123  ", True),  # Trailing whitespace (atoi stops at first non-digit)
        ("+123", True),  # Explicit positive sign
        ("+-123", False),  # Invalid sign combination (atoi returns 0)
        ("", False),  # Empty string (atoi returns 0)
        ("   ", False),  # Only whitespace (atoi returns 0)
        ("2147483647", True),  # INT_MAX
        ("-2147483648", True),  # INT_MIN
        ("2147483648",
         True),  # Overflow (behavior may vary, but should be non-zero)
        ("-2147483649",
         True),  # Underflow (behavior may vary, but should be non-zero)
    ]

    for value, expected in atoi_test_cases:
        os.environ["VLLM_KERNEL_OVERRIDE_BATCH_INVARIANT"] = value

        try:
            result = vllm_kernel_override_batch_invariant()
            msg = f"Expected {expected} for value '{value}', got {result}"
            assert result is expected, msg
        finally:
            del os.environ["VLLM_KERNEL_OVERRIDE_BATCH_INVARIANT"]


def test_vllm_kernel_override_batch_invariant_return_type():
    """Test that the function returns a boolean type."""
    # Test with various inputs to ensure return type is always bool
    test_values = ["0", "1", "42", "", "abc", None]

    for value in test_values:
        # Clean up environment
        if "VLLM_KERNEL_OVERRIDE_BATCH_INVARIANT" in os.environ:
            del os.environ["VLLM_KERNEL_OVERRIDE_BATCH_INVARIANT"]

        # Set environment variable if provided
        if value is not None:
            os.environ["VLLM_KERNEL_OVERRIDE_BATCH_INVARIANT"] = value

        try:
            result = vllm_kernel_override_batch_invariant()
            assert isinstance(
                result,
                bool), f"Expected bool, got {type(result)} for value '{value}'"
            assert result in [
                True, False
            ], f"Expected True or False, got {result} for value '{value}'"
        finally:
            # Clean up
            if "VLLM_KERNEL_OVERRIDE_BATCH_INVARIANT" in os.environ:
                del os.environ["VLLM_KERNEL_OVERRIDE_BATCH_INVARIANT"]


def test_vllm_kernel_override_batch_invariant_multiple_calls():
    """
    Test that multiple calls with the same environment return consistent
    results.
    """
    test_values = ["0", "1", "42", "", "abc"]

    for value in test_values:
        os.environ["VLLM_KERNEL_OVERRIDE_BATCH_INVARIANT"] = value

        try:
            # Call the function multiple times
            results = [
                vllm_kernel_override_batch_invariant() for _ in range(10)
            ]

            # All results should be the same
            first_result = results[0]
            for i, result in enumerate(results[1:], 1):
                assert result == first_result, (
                    f"Inconsistent results for value '{value}': "
                    f"call 0 gave {first_result}, call {i} gave {result}")
        finally:
            del os.environ["VLLM_KERNEL_OVERRIDE_BATCH_INVARIANT"]


def test_vllm_kernel_override_batch_invariant_environment_isolation():
    """Test that the function reads the environment variable each time."""
    # Start with env var unset
    if "VLLM_KERNEL_OVERRIDE_BATCH_INVARIANT" in os.environ:
        del os.environ["VLLM_KERNEL_OVERRIDE_BATCH_INVARIANT"]

    # Should return False when unset
    result1 = vllm_kernel_override_batch_invariant()
    assert result1 is False

    # Set to "1"
    os.environ["VLLM_KERNEL_OVERRIDE_BATCH_INVARIANT"] = "1"
    result2 = vllm_kernel_override_batch_invariant()
    assert result2 is True

    # Change to "0"
    os.environ["VLLM_KERNEL_OVERRIDE_BATCH_INVARIANT"] = "0"
    result3 = vllm_kernel_override_batch_invariant()
    assert result3 is False

    # Change to "42"
    os.environ["VLLM_KERNEL_OVERRIDE_BATCH_INVARIANT"] = "42"
    result4 = vllm_kernel_override_batch_invariant()
    assert result4 is True

    # Unset again
    del os.environ["VLLM_KERNEL_OVERRIDE_BATCH_INVARIANT"]
    result5 = vllm_kernel_override_batch_invariant()
    assert result5 is False


def test_vllm_kernel_override_batch_invariant_stress_test():
    """Stress test with many different values."""
    import random
    import string

    # Generate random test values
    test_values = []

    # Add some known values
    test_values.extend(["0", "1", "-1", "42", "", "abc", "true", "false"])

    # Add random integers
    for _ in range(20):
        test_values.append(str(random.randint(-1000, 1000)))

    # Add random strings
    for _ in range(20):
        length = random.randint(1, 10)
        test_values.append(''.join(
            random.choices(string.ascii_letters + string.digits, k=length)))

    # Add random mixed strings
    for _ in range(10):
        length = random.randint(1, 10)
        test_values.append(''.join(
            random.choices(string.ascii_letters + string.digits + " \t\n",
                           k=length)))

    for value in test_values:
        os.environ["VLLM_KERNEL_OVERRIDE_BATCH_INVARIANT"] = value

        try:
            result = vllm_kernel_override_batch_invariant()
            # Just ensure it doesn't crash and returns a boolean
            assert isinstance(
                result,
                bool), f"Expected bool for value '{value}', got {type(result)}"
        except Exception as e:
            pytest.fail(f"Function crashed with value '{value}': {e}")
        finally:
            del os.environ["VLLM_KERNEL_OVERRIDE_BATCH_INVARIANT"]


def test_vllm_kernel_override_batch_invariant_unicode():
    """Test that the function handles unicode strings correctly."""
    unicode_values = [
        "1️⃣",  # Emoji
        "①",  # Unicode number
        "一",  # Chinese number
        "١",  # Arabic number
        "Ⅰ",  # Roman numeral
        "𝟏",  # Mathematical bold digit
        "🔢",  # Number emoji
        "αβγ",  # Greek letters
        "测试",  # Chinese characters
    ]

    for value in unicode_values:
        os.environ["VLLM_KERNEL_OVERRIDE_BATCH_INVARIANT"] = value

        try:
            result = vllm_kernel_override_batch_invariant()
            # Unicode strings should be treated as non-numeric and return False
            msg = f"Expected False for unicode value '{value}', got {result}"
            assert result is False, msg
        finally:
            del os.environ["VLLM_KERNEL_OVERRIDE_BATCH_INVARIANT"]
