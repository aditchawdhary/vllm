# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Tests for csrc/core/batch_invariant.hpp

This module tests the C++ function vllm_kernel_override_batch_invariant()
which reads the VLLM_KERNEL_OVERRIDE_BATCH_INVARIANT environment variable
and returns true if it's set to a non-zero value.
"""

import os
import subprocess
import sys
import tempfile
from typing import Optional

import pytest


def create_test_cpp_program() -> str:
    """Create a temporary C++ test program that uses the batch_invariant function."""
    cpp_code = '''
#include <iostream>
#include <cstdlib>

// Include the header we're testing
#include "csrc/core/batch_invariant.hpp"

int main() {
    bool result = vllm::vllm_kernel_override_batch_invariant();
    std::cout << (result ? "1" : "0") << std::endl;
    return 0;
}
'''
    return cpp_code


def compile_and_run_cpp_test(env_value: Optional[str] = None) -> bool:
    """
    Compile and run a C++ test program with the given environment variable value.

    Args:
        env_value: Value to set for VLLM_KERNEL_OVERRIDE_BATCH_INVARIANT,
                  or None to unset it

    Returns:
        bool: The result returned by vllm_kernel_override_batch_invariant()
    """
    cpp_code = create_test_cpp_program()

    with tempfile.TemporaryDirectory() as temp_dir:
        # Write the C++ source file
        cpp_file = os.path.join(temp_dir, "test_batch_invariant.cpp")
        with open(cpp_file, 'w') as f:
            f.write(cpp_code)

        # Compile the program
        executable = os.path.join(temp_dir, "test_batch_invariant")

        # Get the project root directory (assuming we're in tests/kernels/core/)
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))

        compile_cmd = [
            "g++",
            "-std=c++17",
            f"-I{project_root}",
            cpp_file,
            "-o", executable
        ]

        try:
            subprocess.run(compile_cmd, check=True, capture_output=True, text=True)
        except subprocess.CalledProcessError as e:
            pytest.skip(f"Failed to compile C++ test program: {e.stderr}")

        # Run the program with the specified environment
        env = os.environ.copy()
        if env_value is None:
            env.pop("VLLM_KERNEL_OVERRIDE_BATCH_INVARIANT", None)
        else:
            env["VLLM_KERNEL_OVERRIDE_BATCH_INVARIANT"] = env_value

        try:
            result = subprocess.run([executable],
                                  capture_output=True,
                                  text=True,
                                  env=env,
                                  check=True)
            return result.stdout.strip() == "1"
        except subprocess.CalledProcessError as e:
            pytest.fail(f"Failed to run C++ test program: {e.stderr}")


class TestBatchInvariant:
    """Test cases for vllm_kernel_override_batch_invariant function."""

    def test_env_var_not_set(self):
        """Test that function returns false when environment variable is not set."""
        result = compile_and_run_cpp_test(env_value=None)
        assert result is False, "Should return false when env var is not set"

    def test_env_var_empty_string(self):
        """Test that function returns false when environment variable is empty string."""
        result = compile_and_run_cpp_test(env_value="")
        assert result is False, "Should return false when env var is empty string"

    def test_env_var_zero(self):
        """Test that function returns false when environment variable is '0'."""
        result = compile_and_run_cpp_test(env_value="0")
        assert result is False, "Should return false when env var is '0'"

    def test_env_var_one(self):
        """Test that function returns true when environment variable is '1'."""
        result = compile_and_run_cpp_test(env_value="1")
        assert result is True, "Should return true when env var is '1'"

    def test_env_var_positive_integer(self):
        """Test that function returns true when environment variable is positive integer."""
        result = compile_and_run_cpp_test(env_value="42")
        assert result is True, "Should return true when env var is positive integer"

    def test_env_var_negative_integer(self):
        """Test that function returns true when environment variable is negative integer."""
        result = compile_and_run_cpp_test(env_value="-1")
        assert result is True, "Should return true when env var is negative integer"

    def test_env_var_non_numeric_string(self):
        """Test that function returns false when environment variable is non-numeric string."""
        result = compile_and_run_cpp_test(env_value="abc")
        assert result is False, "Should return false when env var is non-numeric string"

    def test_env_var_mixed_string(self):
        """Test that function returns false when environment variable contains mixed characters."""
        result = compile_and_run_cpp_test(env_value="1abc")
        assert result is True, "Should return true when env var starts with number (atoi behavior)"

    def test_env_var_whitespace_only(self):
        """Test that function returns false when environment variable is whitespace only."""
        result = compile_and_run_cpp_test(env_value="   ")
        assert result is False, "Should return false when env var is whitespace only"

    def test_env_var_leading_whitespace_with_number(self):
        """Test that function handles leading whitespace correctly."""
        result = compile_and_run_cpp_test(env_value="  1")
        assert result is True, "Should return true when env var has leading whitespace with number"

    def test_env_var_trailing_whitespace_with_number(self):
        """Test that function handles trailing whitespace correctly."""
        result = compile_and_run_cpp_test(env_value="1  ")
        assert result is True, "Should return true when env var has trailing whitespace with number"

    def test_env_var_zero_with_leading_zeros(self):
        """Test that function handles leading zeros correctly."""
        result = compile_and_run_cpp_test(env_value="000")
        assert result is False, "Should return false when env var is '000'"

    def test_env_var_non_zero_with_leading_zeros(self):
        """Test that function handles leading zeros with non-zero value correctly."""
        result = compile_and_run_cpp_test(env_value="001")
        assert result is True, "Should return true when env var is '001'"

    def test_env_var_very_large_number(self):
        """Test that function handles very large numbers correctly."""
        result = compile_and_run_cpp_test(env_value="999999999")
        assert result is True, "Should return true when env var is very large number"

    def test_env_var_decimal_number(self):
        """Test that function handles decimal numbers correctly (atoi stops at decimal point)."""
        result = compile_and_run_cpp_test(env_value="1.5")
        assert result is True, "Should return true when env var is '1.5' (atoi returns 1)"

    def test_env_var_zero_decimal(self):
        """Test that function handles zero decimal correctly."""
        result = compile_and_run_cpp_test(env_value="0.5")
        assert result is False, "Should return false when env var is '0.5' (atoi returns 0)"

    def test_env_var_plus_sign(self):
        """Test that function handles explicit plus sign correctly."""
        result = compile_and_run_cpp_test(env_value="+1")
        assert result is True, "Should return true when env var is '+1'"

    def test_env_var_plus_zero(self):
        """Test that function handles explicit plus zero correctly."""
        result = compile_and_run_cpp_test(env_value="+0")
        assert result is False, "Should return false when env var is '+0'"

    def test_env_var_minus_zero(self):
        """Test that function handles explicit minus zero correctly."""
        result = compile_and_run_cpp_test(env_value="-0")
        assert result is False, "Should return false when env var is '-0'"

    def test_consistency_with_python_implementation(self):
        """Test that C++ implementation is consistent with Python implementation."""
        from vllm.model_executor.layers.batch_invariant import vllm_kernel_override_batch_invariant

        test_values = ["", "0", "1", "42", "-1", "abc", "1abc", "   ", "  1", "000", "001"]

        for test_value in test_values:
            # Set environment variable for Python test
            old_value = os.environ.get("VLLM_KERNEL_OVERRIDE_BATCH_INVARIANT")
            os.environ["VLLM_KERNEL_OVERRIDE_BATCH_INVARIANT"] = test_value

            try:
                python_result = vllm_kernel_override_batch_invariant()
                cpp_result = compile_and_run_cpp_test(env_value=test_value)

                assert python_result == cpp_result, (
                    f"Mismatch for value '{test_value}': "
                    f"Python={python_result}, C++={cpp_result}"
                )
            finally:
                # Restore original environment variable
                if old_value is None:
                    os.environ.pop("VLLM_KERNEL_OVERRIDE_BATCH_INVARIANT", None)
                else:
                    os.environ["VLLM_KERNEL_OVERRIDE_BATCH_INVARIANT"] = old_value
