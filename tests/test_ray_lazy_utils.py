# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import sys
from unittest.mock import MagicMock, patch

import pytest

from vllm.ray.lazy_utils import is_in_ray_actor, is_ray_initialized


class TestIsRayInitialized:
    """Test cases for is_ray_initialized function."""

    @patch('vllm.ray.lazy_utils.ray')
    def test_ray_initialized_true(self, mock_ray):
        """Test when Ray is available and initialized."""
        mock_ray.is_initialized.return_value = True

        result = is_ray_initialized()

        assert result is True
        mock_ray.is_initialized.assert_called_once()

    @patch('vllm.ray.lazy_utils.ray')
    def test_ray_initialized_false(self, mock_ray):
        """Test when Ray is available but not initialized."""
        mock_ray.is_initialized.return_value = False

        result = is_ray_initialized()

        assert result is False
        mock_ray.is_initialized.assert_called_once()

    def test_ray_import_error(self):
        """Test when Ray is not available (ImportError)."""
        # Remove ray from sys.modules if it exists
        original_ray = sys.modules.get('ray')
        if 'ray' in sys.modules:
            del sys.modules['ray']

        # Mock the import to raise ImportError
        with patch.dict('sys.modules', {'ray': None}):
            result = is_ray_initialized()

        assert result is False

        # Restore original ray module if it existed
        if original_ray is not None:
            sys.modules['ray'] = original_ray

    @patch('builtins.__import__')
    def test_ray_import_error_explicit(self, mock_import):
        """Test ImportError handling with explicit mock."""
        def side_effect(name, *args, **kwargs):
            if name == 'ray':
                raise ImportError("No module named 'ray'")
            return __import__(name, *args, **kwargs)

        mock_import.side_effect = side_effect

        result = is_ray_initialized()

        assert result is False

    @patch('vllm.ray.lazy_utils.ray')
    def test_ray_is_initialized_exception(self, mock_ray):
        """Test when ray.is_initialized() raises an exception."""
        mock_ray.is_initialized.side_effect = RuntimeError("Ray error")

        # The function should still handle this gracefully
        with pytest.raises(RuntimeError):
            is_ray_initialized()

    @patch('vllm.ray.lazy_utils.ray')
    def test_ray_is_initialized_none_return(self, mock_ray):
        """Test when ray.is_initialized() returns None."""
        mock_ray.is_initialized.return_value = None

        result = is_ray_initialized()

        assert result is None
        mock_ray.is_initialized.assert_called_once()


class TestIsInRayActor:
    """Test cases for is_in_ray_actor function."""

    @patch('vllm.ray.lazy_utils.ray')
    def test_in_ray_actor_true(self, mock_ray):
        """Test when Ray is initialized and we are in an actor."""
        mock_ray.is_initialized.return_value = True
        mock_runtime_context = MagicMock()
        mock_runtime_context.get_actor_id.return_value = "actor_123"
        mock_ray.get_runtime_context.return_value = mock_runtime_context

        result = is_in_ray_actor()

        assert result is True
        mock_ray.is_initialized.assert_called_once()
        mock_ray.get_runtime_context.assert_called_once()
        mock_runtime_context.get_actor_id.assert_called_once()

    @patch('vllm.ray.lazy_utils.ray')
    def test_in_ray_actor_false_not_initialized(self, mock_ray):
        """Test when Ray is not initialized."""
        mock_ray.is_initialized.return_value = False

        result = is_in_ray_actor()

        assert result is False
        mock_ray.is_initialized.assert_called_once()
        # get_runtime_context should not be called if ray is not initialized
        mock_ray.get_runtime_context.assert_not_called()

    @patch('vllm.ray.lazy_utils.ray')
    def test_in_ray_actor_false_no_actor_id(self, mock_ray):
        """Test when Ray is initialized but we are not in an actor."""
        mock_ray.is_initialized.return_value = True
        mock_runtime_context = MagicMock()
        mock_runtime_context.get_actor_id.return_value = None
        mock_ray.get_runtime_context.return_value = mock_runtime_context

        result = is_in_ray_actor()

        assert result is False
        mock_ray.is_initialized.assert_called_once()
        mock_ray.get_runtime_context.assert_called_once()
        mock_runtime_context.get_actor_id.assert_called_once()

    def test_in_ray_actor_import_error(self):
        """Test when Ray is not available (ImportError)."""
        # Remove ray from sys.modules if it exists
        original_ray = sys.modules.get('ray')
        if 'ray' in sys.modules:
            del sys.modules['ray']

        # Mock the import to raise ImportError
        with patch.dict('sys.modules', {'ray': None}):
            result = is_in_ray_actor()

        assert result is False

        # Restore original ray module if it existed
        if original_ray is not None:
            sys.modules['ray'] = original_ray

    @patch('builtins.__import__')
    def test_in_ray_actor_import_error_explicit(self, mock_import):
        """Test ImportError handling with explicit mock."""
        def side_effect(name, *args, **kwargs):
            if name == 'ray':
                raise ImportError("No module named 'ray'")
            return __import__(name, *args, **kwargs)

        mock_import.side_effect = side_effect

        result = is_in_ray_actor()

        assert result is False

    @patch('vllm.ray.lazy_utils.ray')
    def test_in_ray_actor_runtime_context_exception(self, mock_ray):
        """Test when get_runtime_context() raises an exception."""
        mock_ray.is_initialized.return_value = True
        mock_ray.get_runtime_context.side_effect = RuntimeError("Runtime error")

        # The function should still handle this gracefully
        with pytest.raises(RuntimeError):
            is_in_ray_actor()

    @patch('vllm.ray.lazy_utils.ray')
    def test_in_ray_actor_get_actor_id_exception(self, mock_ray):
        """Test when get_actor_id() raises an exception."""
        mock_ray.is_initialized.return_value = True
        mock_runtime_context = MagicMock()
        mock_runtime_context.get_actor_id.side_effect = RuntimeError("Actor ID error")
        mock_ray.get_runtime_context.return_value = mock_runtime_context

        # The function should still handle this gracefully
        with pytest.raises(RuntimeError):
            is_in_ray_actor()

    @patch('vllm.ray.lazy_utils.ray')
    def test_in_ray_actor_empty_string_actor_id(self, mock_ray):
        """Test when actor ID is an empty string."""
        mock_ray.is_initialized.return_value = True
        mock_runtime_context = MagicMock()
        mock_runtime_context.get_actor_id.return_value = ""
        mock_ray.get_runtime_context.return_value = mock_runtime_context

        result = is_in_ray_actor()

        # Empty string is truthy in Python, so this should return True
        assert result is True
        mock_ray.is_initialized.assert_called_once()
        mock_ray.get_runtime_context.assert_called_once()
        mock_runtime_context.get_actor_id.assert_called_once()
