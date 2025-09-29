# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Tests for csrc/core/batch_invariant.hpp

This module tests the vllm_kernel_override_batch_invariant() function
which checks the VLLM_KERNEL_OVERRIDE_BATCH_INVARIANT environment variable.
"""

import os

import torch

import pytest


def test_vllm_kernel_override_batch_invariant_default():
    """Test that the function returns False when env var is not set."""
    # Ensure the environment variable is not set
    if "VLLM_KERNEL_OVERRIDE_BATCH_INVARIANT" in os.environ:
        del os.environ["VLLM_KERNEL_OVERRIDE_BATCH_INVARIANT"]

    # Call the C++ function through PyTorch binding
    result = torch.ops._C.vllm_kernel_override_batch_invariant()
    assert result is False


def test_vllm_kernel_override_batch_invariant_zero():
    """Test that the function returns False when env var is set to '0'."""
    os.environ["VLLM_KERNEL_OVERRIDE_BATCH_INVARIANT"] = "0"

    try:
        result = torch.ops._C.vllm_kernel_override_batch_invariant()
        assert result is False
    finally:
        del os.environ["VLLM_KERNEL_OVERRIDE_BATCH_INVARIANT"]


def test_vllm_kernel_override_batch_invariant_one():
    """Test that the function returns True when env var is set to '1'."""
    os.environ["VLLM_KERNEL_OVERRIDE_BATCH_INVARIANT"] = "1"

    try:
        result = torch.ops._C.vllm_kernel_override_batch_invariant()
        assert result is True
    finally:
        del os.environ["VLLM_KERNEL_OVERRIDE_BATCH_INVARIANT"]


def test_vllm_kernel_override_batch_invariant_positive_number():
    """Test that the function returns True when env var is set to positive number."""
    os.environ["VLLM_KERNEL_OVERRIDE_BATCH_INVARIANT"] = "42"

    try:
        result = torch.ops._C.vllm_kernel_override_batch_invariant()
        assert result is True
    finally:
        del os.environ["VLLM_KERNEL_OVERRIDE_BATCH_INVARIANT"]


def test_vllm_kernel_override_batch_invariant_negative_number():
    """Test that the function returns True when env var is set to negative number."""
    os.environ["VLLM_KERNEL_OVERRIDE_BATCH_INVARIANT"] = "-1"

    try:
        result = torch.ops._C.vllm_kernel_override_batch_invariant()
        assert result is True
    finally:
        del os.environ["VLLM_KERNEL_OVERRIDE_BATCH_INVARIANT"]


def test_vllm_kernel_override_batch_invariant_empty_string():
    """Test that the function returns False when env var is set to empty string."""
    os.environ["VLLM_KERNEL_OVERRIDE_BATCH_INVARIANT"] = ""

    try:
        result = torch.ops._C.vllm_kernel_override_batch_invariant()
        assert result is False
    finally:
        del os.environ["VLLM_KERNEL_OVERRIDE_BATCH_INVARIANT"]


def test_vllm_kernel_override_batch_invariant_non_numeric():
    """Test that the function returns False when env var is set to non-numeric string."""
    test_values = ["abc", "true", "false", "yes", "no", "on", "off"]

    for value in test_values:
        os.environ["VLLM_KERNEL_OVERRIDE_BATCH_INVARIANT"] = value

        try:
            result = torch.ops._C.vllm_kernel_override_batch_invariant()
            assert result is False, f"Expected False for value '{value}', got {result}"
        finally:
            del os.environ["VLLM_KERNEL_OVERRIDE_BATCH_INVARIANT"]


def test_vllm_kernel_override_batch_invariant_whitespace():
    """Test that the function handles whitespace correctly."""
    test_values = [" 1 ", "\t1\t", "\n1\n", " 0 ", "\t0\t", "\n0\n"]
    expected_results = [True, True, True, False, False, False]

    for value, expected in zip(test_values, expected_results):
        os.environ["VLLM_KERNEL_OVERRIDE_BATCH_INVARIANT"] = value

        try:
            result = torch.ops._C.vllm_kernel_override_batch_invariant()
            assert result is expected, f"Expected {expected} for value '{value}', got {result}"
        finally:
            del os.environ["VLLM_KERNEL_OVERRIDE_BATCH_INVARIANT"]


def test_vllm_kernel_override_batch_invariant_mixed_numeric():
    """Test that the function handles mixed numeric strings correctly."""
    test_values = ["1abc", "abc1", "1.0", "1.5", "0.0", "0.5"]
    expected_results = [True, False, True, True, False, False]

    for value, expected in zip(test_values, expected_results):
        os.environ["VLLM_KERNEL_OVERRIDE_BATCH_INVARIANT"] = value

        try:
            result = torch.ops._C.vllm_kernel_override_batch_invariant()
            assert result is expected, f"Expected {expected} for value '{value}', got {result}"
        finally:
            del os.environ["VLLM_KERNEL_OVERRIDE_BATCH_INVARIANT"]


def test_vllm_kernel_override_batch_invariant_large_numbers():
    """Test that the function handles large numbers correctly."""
    test_values = ["999999999", "-999999999", "2147483647", "-2147483648"]
    expected_results = [True, True, True, True]

    for value, expected in zip(test_values, expected_results):
        os.environ["VLLM_KERNEL_OVERRIDE_BATCH_INVARIANT"] = value

        try:
            result = torch.ops._C.vllm_kernel_override_batch_invariant()
            assert result is expected, f"Expected {expected} for value '{value}', got {result}"
        finally:
            del os.environ["VLLM_KERNEL_OVERRIDE_BATCH_INVARIANT"]


@pytest.mark.parametrize("env_value,expected", [
    (None, False),  # Environment variable not set
    ("", False),    # Empty string
    ("0", False),   # Zero
    ("1", True),    # One
    ("42", True),   # Positive number
    ("-1", True),   # Negative number
    ("abc", False), # Non-numeric string
    ("1.0", True),  # Float that converts to 1
    ("0.0", False), # Float that converts to 0
    (" 1 ", True),  # Whitespace around 1
    (" 0 ", False), # Whitespace around 0
])
def test_vllm_kernel_override_batch_invariant_parametrized(env_value, expected):
    """Parametrized test for various environment variable values."""
    # Clean up environment
    if "VLLM_KERNEL_OVERRIDE_BATCH_INVARIANT" in os.environ:
        del os.environ["VLLM_KERNEL_OVERRIDE_BATCH_INVARIANT"]

    # Set environment variable if provided
    if env_value is not None:
        os.environ["VLLM_KERNEL_OVERRIDE_BATCH_INVARIANT"] = env_value

    try:
        result = torch.ops._C.vllm_kernel_override_batch_invariant()
        assert result is expected, f"Expected {expected} for env_value '{env_value}', got {result}"
    finally:
        # Clean up
        if "VLLM_KERNEL_OVERRIDE_BATCH_INVARIANT" in os.environ:
            del os.environ["VLLM_KERNEL_OVERRIDE_BATCH_INVARIANT"]


def test_vllm_kernel_override_batch_invariant_consistency_with_python():
    """Test that C++ function behavior matches Python implementation."""
    from vllm.model_executor.layers.batch_invariant import \
        vllm_kernel_override_batch_invariant as python_impl

    test_values = [
        None, "", "0", "1", "42", "-1", "abc", "1.0", "0.0",
        " 1 ", " 0 ", "true", "false", "999999999"
    ]

    for value in test_values:
        # Clean up environment
        if "VLLM_KERNEL_OVERRIDE_BATCH_INVARIANT" in os.environ:
            del os.environ["VLLM_KERNEL_OVERRIDE_BATCH_INVARIANT"]

        # Set environment variable if provided
        if value is not None:
            os.environ["VLLM_KERNEL_OVERRIDE_BATCH_INVARIANT"] = value

        try:
            cpp_result = torch.ops._C.vllm_kernel_override_batch_invariant()
            python_result = python_impl()

            assert cpp_result == python_result, (
                f"C++ and Python implementations differ for value '{value}': "
                f"C++ returned {cpp_result}, Python returned {python_result}"
            )
        finally:
            # Clean up
            if "VLLM_KERNEL_OVERRIDE_BATCH_INVARIANT" in os.environ:
                del os.environ["VLLM_KERNEL_OVERRIDE_BATCH_INVARIANT"]


def test_vllm_kernel_override_batch_invariant_thread_safety():
    """Test that the function is thread-safe (basic test)."""
    import threading
    import time

    results = []
    errors = []

    def worker(env_value, expected):
        try:
            os.environ["VLLM_KERNEL_OVERRIDE_BATCH_INVARIANT"] = env_value
            time.sleep(0.01)  # Small delay to increase chance of race conditions
            result = torch.ops._C.vllm_kernel_override_batch_invariant()
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
    assert len(results) == len(threads), "Not all threads completed successfully"


def test_vllm_kernel_override_batch_invariant_edge_cases():
    """Test edge cases and corner cases."""
    edge_cases = [
        ("00", False),      # Leading zero
        ("01", True),       # Leading zero with non-zero
        ("+1", True),       # Positive sign
        ("+0", False),      # Positive sign with zero
        ("1e0", True),      # Scientific notation
        ("0e0", False),     # Scientific notation zero
        ("inf", False),     # Infinity string
        ("nan", False),     # NaN string
        ("0x1", False),     # Hexadecimal (should be treated as non-numeric)
        ("0b1", False),     # Binary (should be treated as non-numeric)
    ]

    for value, expected in edge_cases:
        os.environ["VLLM_KERNEL_OVERRIDE_BATCH_INVARIANT"] = value

        try:
            result = torch.ops._C.vllm_kernel_override_batch_invariant()
            assert result is expected, f"Expected {expected} for value '{value}', got {result}"
        finally:
            del os.environ["VLLM_KERNEL_OVERRIDE_BATCH_INVARIANT"]
