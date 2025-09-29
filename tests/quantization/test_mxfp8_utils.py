# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for vllm/model_executor/layers/quantization/utils/mxfp8_utils.py"""

import sys
from unittest.mock import MagicMock, patch

import torch
from vllm.model_executor.layers.quantization.utils.mxfp8_utils import \
    mxfp8_quantize

import pytest


class TestMxfp8Quantize:
    """Test cases for mxfp8_quantize function."""

    @pytest.fixture
    def sample_tensor(self):
        """Create a sample tensor for testing."""
        return torch.randn(4, 8, dtype=torch.float32)

    @pytest.fixture
    def mock_flashinfer_success(self):
        """Mock flashinfer module with successful quantization."""
        mock_quantize = MagicMock()
        mock_quantize.return_value = (
            torch.randn(4, 8, dtype=torch.float8_e4m3fn),
            torch.randn(4, 1, dtype=torch.float32)
        )

        mock_module = MagicMock()
        mock_module.mxfp8_quantize = mock_quantize

        with patch.dict('sys.modules', {'flashinfer': mock_module}):
            yield mock_quantize

    @pytest.fixture
    def mock_flashinfer_import_error(self):
        """Mock flashinfer module to raise ImportError."""
        with patch.dict('sys.modules', {'flashinfer': None}):
            with patch('builtins.__import__', side_effect=ImportError("No module named 'flashinfer'")):
                yield

    def test_mxfp8_quantize_success(self, sample_tensor, mock_flashinfer_success):
        """Test successful quantization when flashinfer is available."""
        result = mxfp8_quantize(sample_tensor)

        # Verify the function returns a tuple of two tensors
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], torch.Tensor)
        assert isinstance(result[1], torch.Tensor)

        # Verify flashinfer.mxfp8_quantize was called with correct parameters
        mock_flashinfer_success.assert_called_once_with(
            sample_tensor, is_sf_swizzled_layout=False
        )

    def test_mxfp8_quantize_import_error(self, sample_tensor, mock_flashinfer_import_error):
        """Test ImportError when flashinfer is not available."""
        with pytest.raises(ImportError) as exc_info:
            mxfp8_quantize(sample_tensor)

        # Verify the error message contains the expected information
        error_msg = str(exc_info.value)
        assert "The package `flashinfer` is required to do MX-FP8 quantization" in error_msg
        assert "pip install flashinfer" in error_msg

        # Verify the original ImportError is chained
        assert exc_info.value.__cause__ is not None
        assert isinstance(exc_info.value.__cause__, ImportError)

    def test_mxfp8_quantize_different_tensor_shapes(self, mock_flashinfer_success):
        """Test quantization with different tensor shapes."""
        test_cases = [
            torch.randn(2, 4),      # 2D tensor
            torch.randn(1, 1, 8),   # 3D tensor
            torch.randn(2, 3, 4, 5), # 4D tensor
        ]

        for tensor in test_cases:
            result = mxfp8_quantize(tensor)
            assert isinstance(result, tuple)
            assert len(result) == 2
            mock_flashinfer_success.assert_called_with(
                tensor, is_sf_swizzled_layout=False
            )

    def test_mxfp8_quantize_different_dtypes(self, mock_flashinfer_success):
        """Test quantization with different tensor dtypes."""
        test_cases = [
            torch.randn(4, 8, dtype=torch.float16),
            torch.randn(4, 8, dtype=torch.float32),
            torch.randn(4, 8, dtype=torch.float64),
            torch.randn(4, 8, dtype=torch.bfloat16),
        ]

        for tensor in test_cases:
            result = mxfp8_quantize(tensor)
            assert isinstance(result, tuple)
            assert len(result) == 2
            mock_flashinfer_success.assert_called_with(
                tensor, is_sf_swizzled_layout=False
            )

    def test_mxfp8_quantize_edge_cases(self, mock_flashinfer_success):
        """Test quantization with edge case tensors."""
        # Single element tensor
        single_element = torch.tensor([[1.0]])
        result = mxfp8_quantize(single_element)
        assert isinstance(result, tuple)
        assert len(result) == 2

        # Large tensor
        large_tensor = torch.randn(100, 200)
        result = mxfp8_quantize(large_tensor)
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_mxfp8_quantize_zero_tensor(self, mock_flashinfer_success):
        """Test quantization with zero tensor."""
        zero_tensor = torch.zeros(4, 8)
        result = mxfp8_quantize(zero_tensor)
        assert isinstance(result, tuple)
        assert len(result) == 2
        mock_flashinfer_success.assert_called_with(
            zero_tensor, is_sf_swizzled_layout=False
        )

    def test_mxfp8_quantize_ones_tensor(self, mock_flashinfer_success):
        """Test quantization with ones tensor."""
        ones_tensor = torch.ones(4, 8)
        result = mxfp8_quantize(ones_tensor)
        assert isinstance(result, tuple)
        assert len(result) == 2
        mock_flashinfer_success.assert_called_with(
            ones_tensor, is_sf_swizzled_layout=False
        )

    def test_mxfp8_quantize_parameter_passing(self, sample_tensor, mock_flashinfer_success):
        """Test that the is_sf_swizzled_layout parameter is correctly passed."""
        mxfp8_quantize(sample_tensor)

        # Verify the parameter is always False as hardcoded in the function
        mock_flashinfer_success.assert_called_once_with(
            sample_tensor, is_sf_swizzled_layout=False
        )

    def test_mxfp8_quantize_return_value_structure(self, sample_tensor, mock_flashinfer_success):
        """Test that the return value structure is preserved from flashinfer."""
        # Set up specific return values
        expected_quantized = torch.randn(4, 8, dtype=torch.float8_e4m3fn)
        expected_scale = torch.randn(4, 1, dtype=torch.float32)
        mock_flashinfer_success.return_value = (expected_quantized, expected_scale)

        result = mxfp8_quantize(sample_tensor)

        # Verify the exact return values are passed through
        assert result[0] is expected_quantized
        assert result[1] is expected_scale
