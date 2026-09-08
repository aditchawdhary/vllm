# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Simple smoke test for batch_invariant functionality.

This is a minimal test to ensure the basic functionality works.
"""

import os

import pytest


def test_batch_invariant_import():
    """Test that we can import the function."""
    try:
        from vllm.model_executor.layers.batch_invariant import (
            vllm_kernel_override_batch_invariant)
        assert callable(vllm_kernel_override_batch_invariant)
    except ImportError:
        pytest.skip("vllm.model_executor.layers.batch_invariant not available")


def test_batch_invariant_basic():
    """Basic test of the batch invariant function."""
    try:
        from vllm.model_executor.layers.batch_invariant import (
            vllm_kernel_override_batch_invariant)
    except ImportError:
        pytest.skip("vllm.model_executor.layers.batch_invariant not available")

    # Clean environment
    if "VLLM_KERNEL_OVERRIDE_BATCH_INVARIANT" in os.environ:
        del os.environ["VLLM_KERNEL_OVERRIDE_BATCH_INVARIANT"]

    # Test default (should be False)
    result = vllm_kernel_override_batch_invariant()
    assert result is False

    # Test with "1" (should be True)
    os.environ["VLLM_KERNEL_OVERRIDE_BATCH_INVARIANT"] = "1"
    try:
        result = vllm_kernel_override_batch_invariant()
        assert result is True
    finally:
        del os.environ["VLLM_KERNEL_OVERRIDE_BATCH_INVARIANT"]

    # Test with "0" (should be False)
    os.environ["VLLM_KERNEL_OVERRIDE_BATCH_INVARIANT"] = "0"
    try:
        result = vllm_kernel_override_batch_invariant()
        assert result is False
    finally:
        del os.environ["VLLM_KERNEL_OVERRIDE_BATCH_INVARIANT"]
