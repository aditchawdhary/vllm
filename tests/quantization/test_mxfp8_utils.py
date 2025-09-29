# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for vllm/model_executor/layers/quantization/utils/mxfp8_utils.py

Run `pytest tests/quantization/test_mxfp8_utils.py --forked`.
"""
import sys
from unittest.mock import MagicMock, patch

import pytest
import torch

from vllm.model_executor.layers.quantization.utils.mxfp8_utils import (
    mxfp8_quantize)


class TestMxfp8Quantize:
    """Test class for mxfp8_quantize function."""

    def test_mxfp8_quantize_import_error(self):
        """Test that ImportError is raised when flashinfer is not available."""
        # Mock the import to raise ImportError
        with patch.dict('sys.modules', {'flashinfer': None}):
            with patch('builtins.__import__', side_effect=ImportError("No module named 'flashinfer'")):
                # Create a test tensor
                x = torch.randn(2, 3, dtype=torch.float32)

                # Test that ImportError is raised with the correct message
                with pytest.raises(ImportError) as exc_info:
                    mxfp8_quantize(x)

                # Verify the error message
                expected_msg = ("The package `flashinfer` is required to do "
                              "MX-FP8 quantization. Please install it with"
                              "`pip install flashinfer`")
                assert expected_msg in str(exc_info.value)

                # Verify the original error is chained
                assert exc_info.value.__cause__ is not None
                assert "No module named 'flashinfer'" in str(exc_info.value.__cause__)

    def test_mxfp8_quantize_success_basic(self):
        """Test successful quantization with basic tensor."""
        # Mock flashinfer module and its mxfp8_quantize function
        mock_flashinfer = MagicMock()
        mock_quantize_result = (torch.randn(2, 3), torch.randn(2, 1))
        mock_flashinfer.mxfp8_quantize.return_value = mock_quantize_result

        with patch.dict('sys.modules', {'flashinfer': mock_flashinfer}):
            # Create a test tensor
            x = torch.randn(2, 3, dtype=torch.float32)

            # Call the function
            result = mxfp8_quantize(x)

            # Verify the result
            assert isinstance(result, tuple)
            assert len(result) == 2
            assert torch.is_tensor(result[0])
            assert torch.is_tensor(result[1])

            # Verify flashinfer.mxfp8_quantize was called with correct parameters
            mock_flashinfer.mxfp8_quantize.assert_called_once_with(
                x, is_sf_swizzled_layout=False
            )

            # Verify the result is what we mocked
            assert torch.equal(result[0], mock_quantize_result[0])
            assert torch.equal(result[1], mock_quantize_result[1])

    def test_mxfp8_quantize_different_tensor_shapes(self):
        """Test quantization with different tensor shapes."""
        # Mock flashinfer module
        mock_flashinfer = MagicMock()

        test_shapes = [
            (1,),           # 1D tensor
            (2, 3),         # 2D tensor
            (2, 3, 4),      # 3D tensor
            (2, 3, 4, 5),   # 4D tensor
            (1, 1),         # Small tensor
            (100, 200),     # Large tensor
        ]

        with patch.dict('sys.modules', {'flashinfer': mock_flashinfer}):
            for shape in test_shapes:
                # Reset mock for each test
                mock_flashinfer.reset_mock()

                # Create mock return value with appropriate shape
                mock_result = (torch.randn(*shape), torch.randn(shape[0], 1) if len(shape) > 1 else torch.randn(1))
                mock_flashinfer.mxfp8_quantize.return_value = mock_result

                # Create test tensor
                x = torch.randn(*shape, dtype=torch.float32)

                # Call function
                result = mxfp8_quantize(x)

                # Verify call was made correctly
                mock_flashinfer.mxfp8_quantize.assert_called_once_with(
                    x, is_sf_swizzled_layout=False
                )

                # Verify result structure
                assert isinstance(result, tuple)
                assert len(result) == 2
                assert torch.is_tensor(result[0])
                assert torch.is_tensor(result[1])

    def test_mxfp8_quantize_different_dtypes(self):
        """Test quantization with different tensor dtypes."""
        # Mock flashinfer module
        mock_flashinfer = MagicMock()
        mock_result = (torch.randn(2, 3), torch.randn(2, 1))
        mock_flashinfer.mxfp8_quantize.return_value = mock_result

        test_dtypes = [
            torch.float32,
            torch.float16,
            torch.bfloat16,
            torch.float64,
        ]

        with patch.dict('sys.modules', {'flashinfer': mock_flashinfer}):
            for dtype in test_dtypes:
                # Reset mock for each test
                mock_flashinfer.reset_mock()

                # Create test tensor with specific dtype
                x = torch.randn(2, 3, dtype=dtype)

                # Call function
                result = mxfp8_quantize(x)

                # Verify call was made correctly
                mock_flashinfer.mxfp8_quantize.assert_called_once_with(
                    x, is_sf_swizzled_layout=False
                )

                # Verify result structure
                assert isinstance(result, tuple)
                assert len(result) == 2

    def test_mxfp8_quantize_empty_tensor(self):
        """Test quantization with empty tensor."""
        # Mock flashinfer module
        mock_flashinfer = MagicMock()
        mock_result = (torch.empty(0, 3), torch.empty(0, 1))
        mock_flashinfer.mxfp8_quantize.return_value = mock_result

        with patch.dict('sys.modules', {'flashinfer': mock_flashinfer}):
            # Create empty tensor
            x = torch.empty(0, 3, dtype=torch.float32)

            # Call function
            result = mxfp8_quantize(x)

            # Verify call was made correctly
            mock_flashinfer.mxfp8_quantize.assert_called_once_with(
                x, is_sf_swizzled_layout=False
            )

            # Verify result structure
            assert isinstance(result, tuple)
            assert len(result) == 2
            assert torch.is_tensor(result[0])
            assert torch.is_tensor(result[1])

    def test_mxfp8_quantize_zero_tensor(self):
        """Test quantization with tensor of zeros."""
        # Mock flashinfer module
        mock_flashinfer = MagicMock()
        mock_result = (torch.zeros(2, 3), torch.zeros(2, 1))
        mock_flashinfer.mxfp8_quantize.return_value = mock_result

        with patch.dict('sys.modules', {'flashinfer': mock_flashinfer}):
            # Create tensor of zeros
            x = torch.zeros(2, 3, dtype=torch.float32)

            # Call function
            result = mxfp8_quantize(x)

            # Verify call was made correctly
            mock_flashinfer.mxfp8_quantize.assert_called_once_with(
                x, is_sf_swizzled_layout=False
            )

            # Verify result structure
            assert isinstance(result, tuple)
            assert len(result) == 2

    def test_mxfp8_quantize_parameter_passing(self):
        """Test that is_sf_swizzled_layout=False is always passed."""
        # Mock flashinfer module
        mock_flashinfer = MagicMock()
        mock_result = (torch.randn(2, 3), torch.randn(2, 1))
        mock_flashinfer.mxfp8_quantize.return_value = mock_result

        with patch.dict('sys.modules', {'flashinfer': mock_flashinfer}):
            x = torch.randn(2, 3, dtype=torch.float32)
            mxfp8_quantize(x)

            # Verify the exact call signature
            mock_flashinfer.mxfp8_quantize.assert_called_once_with(
                x, is_sf_swizzled_layout=False
            )
