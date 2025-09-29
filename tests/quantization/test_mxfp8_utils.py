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

    def test_mxfp8_quantize_success(self, sample_tensor):
        """Test successful quantization when flashinfer is available."""
        # Mock the flashinfer module and its mxfp8_quantize function
        mock_quantize_func = MagicMock()
        expected_result = (
            torch.randn(4, 8, dtype=torch.float8_e4m3fn),
            torch.randn(4, 1, dtype=torch.float32)
        )
        mock_quantize_func.return_value = expected_result

        mock_flashinfer = MagicMock()
        mock_flashinfer.mxfp8_quantize = mock_quantize_func

        with patch.dict('sys.modules', {'flashinfer': mock_flashinfer}):
            result = mxfp8_quantize(sample_tensor)

        # Verify the function returns a tuple of two tensors
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], torch.Tensor)
        assert isinstance(result[1], torch.Tensor)
        assert result == expected_result

        # Verify flashinfer.mxfp8_quantize was called with correct parameters
        mock_quantize_func.assert_called_once_with(
            sample_tensor, is_sf_swizzled_layout=False
        )

    def test_mxfp8_quantize_import_error(self, sample_tensor):
        """Test ImportError when flashinfer is not available."""
        # Remove flashinfer from sys.modules if it exists
        original_modules = sys.modules.copy()
        if 'flashinfer' in sys.modules:
            del sys.modules['flashinfer']

        # Mock the import to raise ImportError
        def mock_import(name, *args, **kwargs):
            if name == 'flashinfer':
                raise ImportError("No module named 'flashinfer'")
            return original_modules.get(name)

        with patch('builtins.__import__', side_effect=mock_import):
            with pytest.raises(ImportError) as exc_info:
                mxfp8_quantize(sample_tensor)

        # Restore original modules
        sys.modules.update(original_modules)

        # Verify the error message contains the expected information
        error_msg = str(exc_info.value)
        assert "The package `flashinfer` is required to do MX-FP8 quantization" in error_msg
        assert "pip install flashinfer" in error_msg

        # Verify the original ImportError is chained
        assert exc_info.value.__cause__ is not None
        assert isinstance(exc_info.value.__cause__, ImportError)

    def test_mxfp8_quantize_different_tensor_shapes(self):
        """Test quantization with different tensor shapes."""
        mock_quantize_func = MagicMock()
        mock_flashinfer = MagicMock()
        mock_flashinfer.mxfp8_quantize = mock_quantize_func

        test_cases = [
            torch.randn(2, 4),      # 2D tensor
            torch.randn(1, 1, 8),   # 3D tensor
            torch.randn(2, 3, 4, 5), # 4D tensor
        ]

        with patch.dict('sys.modules', {'flashinfer': mock_flashinfer}):
            for tensor in test_cases:
                mock_quantize_func.return_value = (
                    torch.randn(*tensor.shape, dtype=torch.float8_e4m3fn),
                    torch.randn(tensor.shape[0], 1, dtype=torch.float32)
                )
                result = mxfp8_quantize(tensor)
                assert isinstance(result, tuple)
                assert len(result) == 2
                mock_quantize_func.assert_called_with(
                    tensor, is_sf_swizzled_layout=False
                )

    def test_mxfp8_quantize_different_dtypes(self):
        """Test quantization with different tensor dtypes."""
        mock_quantize_func = MagicMock()
        mock_flashinfer = MagicMock()
        mock_flashinfer.mxfp8_quantize = mock_quantize_func

        test_cases = [
            torch.randn(4, 8, dtype=torch.float16),
            torch.randn(4, 8, dtype=torch.float32),
            torch.randn(4, 8, dtype=torch.float64),
            torch.randn(4, 8, dtype=torch.bfloat16),
        ]

        with patch.dict('sys.modules', {'flashinfer': mock_flashinfer}):
            for tensor in test_cases:
                mock_quantize_func.return_value = (
                    torch.randn(4, 8, dtype=torch.float8_e4m3fn),
                    torch.randn(4, 1, dtype=torch.float32)
                )
                result = mxfp8_quantize(tensor)
                assert isinstance(result, tuple)
                assert len(result) == 2
                mock_quantize_func.assert_called_with(
                    tensor, is_sf_swizzled_layout=False
                )

    def test_mxfp8_quantize_edge_cases(self):
        """Test quantization with edge case tensors."""
        mock_quantize_func = MagicMock()
        mock_flashinfer = MagicMock()
        mock_flashinfer.mxfp8_quantize = mock_quantize_func

        with patch.dict('sys.modules', {'flashinfer': mock_flashinfer}):
            # Single element tensor
            single_element = torch.tensor([[1.0]])
            mock_quantize_func.return_value = (
                torch.tensor([[1.0]], dtype=torch.float8_e4m3fn),
                torch.tensor([[1.0]], dtype=torch.float32)
            )
            result = mxfp8_quantize(single_element)
            assert isinstance(result, tuple)
            assert len(result) == 2

            # Large tensor
            large_tensor = torch.randn(100, 200)
            mock_quantize_func.return_value = (
                torch.randn(100, 200, dtype=torch.float8_e4m3fn),
                torch.randn(100, 1, dtype=torch.float32)
            )
            result = mxfp8_quantize(large_tensor)
            assert isinstance(result, tuple)
            assert len(result) == 2

    def test_mxfp8_quantize_zero_tensor(self):
        """Test quantization with zero tensor."""
        mock_quantize_func = MagicMock()
        mock_flashinfer = MagicMock()
        mock_flashinfer.mxfp8_quantize = mock_quantize_func

        zero_tensor = torch.zeros(4, 8)
        mock_quantize_func.return_value = (
            torch.zeros(4, 8, dtype=torch.float8_e4m3fn),
            torch.zeros(4, 1, dtype=torch.float32)
        )

        with patch.dict('sys.modules', {'flashinfer': mock_flashinfer}):
            result = mxfp8_quantize(zero_tensor)
            assert isinstance(result, tuple)
            assert len(result) == 2
            mock_quantize_func.assert_called_with(
                zero_tensor, is_sf_swizzled_layout=False
            )

    def test_mxfp8_quantize_ones_tensor(self):
        """Test quantization with ones tensor."""
        mock_quantize_func = MagicMock()
        mock_flashinfer = MagicMock()
        mock_flashinfer.mxfp8_quantize = mock_quantize_func

        ones_tensor = torch.ones(4, 8)
        mock_quantize_func.return_value = (
            torch.ones(4, 8, dtype=torch.float8_e4m3fn),
            torch.ones(4, 1, dtype=torch.float32)
        )

        with patch.dict('sys.modules', {'flashinfer': mock_flashinfer}):
            result = mxfp8_quantize(ones_tensor)
            assert isinstance(result, tuple)
            assert len(result) == 2
            mock_quantize_func.assert_called_with(
                ones_tensor, is_sf_swizzled_layout=False
            )

    def test_mxfp8_quantize_parameter_passing(self, sample_tensor):
        """Test that the is_sf_swizzled_layout parameter is correctly passed."""
        mock_quantize_func = MagicMock()
        mock_flashinfer = MagicMock()
        mock_flashinfer.mxfp8_quantize = mock_quantize_func
        mock_quantize_func.return_value = (
            torch.randn(4, 8, dtype=torch.float8_e4m3fn),
            torch.randn(4, 1, dtype=torch.float32)
        )

        with patch.dict('sys.modules', {'flashinfer': mock_flashinfer}):
            mxfp8_quantize(sample_tensor)

        # Verify the parameter is always False as hardcoded in the function
        mock_quantize_func.assert_called_once_with(
            sample_tensor, is_sf_swizzled_layout=False
        )

    def test_mxfp8_quantize_return_value_structure(self, sample_tensor):
        """Test that the return value structure is preserved from flashinfer."""
        mock_quantize_func = MagicMock()
        mock_flashinfer = MagicMock()
        mock_flashinfer.mxfp8_quantize = mock_quantize_func

        # Set up specific return values
        expected_quantized = torch.randn(4, 8, dtype=torch.float8_e4m3fn)
        expected_scale = torch.randn(4, 1, dtype=torch.float32)
        mock_quantize_func.return_value = (expected_quantized, expected_scale)

        with patch.dict('sys.modules', {'flashinfer': mock_flashinfer}):
            result = mxfp8_quantize(sample_tensor)

        # Verify the exact return values are passed through
        assert result[0] is expected_quantized
        assert result[1] is expected_scale

    def test_mxfp8_quantize_empty_tensor(self):
        """Test quantization with empty tensor."""
        mock_quantize_func = MagicMock()
        mock_flashinfer = MagicMock()
        mock_flashinfer.mxfp8_quantize = mock_quantize_func

        empty_tensor = torch.empty(0, 8)
        mock_quantize_func.return_value = (
            torch.empty(0, 8, dtype=torch.float8_e4m3fn),
            torch.empty(0, 1, dtype=torch.float32)
        )

        with patch.dict('sys.modules', {'flashinfer': mock_flashinfer}):
            result = mxfp8_quantize(empty_tensor)
            assert isinstance(result, tuple)
            assert len(result) == 2
            mock_quantize_func.assert_called_with(
                empty_tensor, is_sf_swizzled_layout=False
            )

    def test_mxfp8_quantize_negative_values(self):
        """Test quantization with tensor containing negative values."""
        mock_quantize_func = MagicMock()
        mock_flashinfer = MagicMock()
        mock_flashinfer.mxfp8_quantize = mock_quantize_func

        negative_tensor = torch.tensor([[-1.0, -2.0], [3.0, -4.0]])
        mock_quantize_func.return_value = (
            torch.tensor([[-1.0, -2.0], [3.0, -4.0]], dtype=torch.float8_e4m3fn),
            torch.tensor([[1.0], [1.0]], dtype=torch.float32)
        )

        with patch.dict('sys.modules', {'flashinfer': mock_flashinfer}):
            result = mxfp8_quantize(negative_tensor)
            assert isinstance(result, tuple)
            assert len(result) == 2
            mock_quantize_func.assert_called_with(
                negative_tensor, is_sf_swizzled_layout=False
            )

    def test_mxfp8_quantize_extreme_values(self):
        """Test quantization with tensor containing extreme values."""
        mock_quantize_func = MagicMock()
        mock_flashinfer = MagicMock()
        mock_flashinfer.mxfp8_quantize = mock_quantize_func

        extreme_tensor = torch.tensor([[float('inf'), float('-inf')],
                                     [1e10, -1e10]])
        mock_quantize_func.return_value = (
            torch.tensor([[1.0, -1.0], [1.0, -1.0]], dtype=torch.float8_e4m3fn),
            torch.tensor([[1.0], [1.0]], dtype=torch.float32)
        )

        with patch.dict('sys.modules', {'flashinfer': mock_flashinfer}):
            result = mxfp8_quantize(extreme_tensor)
            assert isinstance(result, tuple)
            assert len(result) == 2
            mock_quantize_func.assert_called_with(
                extreme_tensor, is_sf_swizzled_layout=False
            )

    def test_mxfp8_quantize_nan_values(self):
        """Test quantization with tensor containing NaN values."""
        mock_quantize_func = MagicMock()
        mock_flashinfer = MagicMock()
        mock_flashinfer.mxfp8_quantize = mock_quantize_func

        nan_tensor = torch.tensor([[float('nan'), 1.0], [2.0, float('nan')]])
        mock_quantize_func.return_value = (
            torch.tensor([[0.0, 1.0], [2.0, 0.0]], dtype=torch.float8_e4m3fn),
            torch.tensor([[1.0], [1.0]], dtype=torch.float32)
        )

        with patch.dict('sys.modules', {'flashinfer': mock_flashinfer}):
            result = mxfp8_quantize(nan_tensor)
            assert isinstance(result, tuple)
            assert len(result) == 2
            mock_quantize_func.assert_called_with(
                nan_tensor, is_sf_swizzled_layout=False
            )

    def test_mxfp8_quantize_1d_tensor(self):
        """Test quantization with 1D tensor."""
        mock_quantize_func = MagicMock()
        mock_flashinfer = MagicMock()
        mock_flashinfer.mxfp8_quantize = mock_quantize_func

        tensor_1d = torch.randn(8)
        mock_quantize_func.return_value = (
            torch.randn(8, dtype=torch.float8_e4m3fn),
            torch.randn(1, dtype=torch.float32)
        )

        with patch.dict('sys.modules', {'flashinfer': mock_flashinfer}):
            result = mxfp8_quantize(tensor_1d)
            assert isinstance(result, tuple)
            assert len(result) == 2
            mock_quantize_func.assert_called_with(
                tensor_1d, is_sf_swizzled_layout=False
            )

    def test_mxfp8_quantize_high_dimensional_tensor(self):
        """Test quantization with high-dimensional tensor."""
        mock_quantize_func = MagicMock()
        mock_flashinfer = MagicMock()
        mock_flashinfer.mxfp8_quantize = mock_quantize_func

        tensor_5d = torch.randn(2, 3, 4, 5, 6)
        mock_quantize_func.return_value = (
            torch.randn(2, 3, 4, 5, 6, dtype=torch.float8_e4m3fn),
            torch.randn(2, 1, 1, 1, 1, dtype=torch.float32)
        )

        with patch.dict('sys.modules', {'flashinfer': mock_flashinfer}):
            result = mxfp8_quantize(tensor_5d)
            assert isinstance(result, tuple)
            assert len(result) == 2
