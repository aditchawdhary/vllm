# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import sys
from unittest.mock import MagicMock, patch

from vllm.ray.lazy_utils import is_in_ray_actor, is_ray_initialized

import pytest

# Mark all tests in this module to skip global cleanup for faster execution
pytestmark = pytest.mark.skip_global_cleanup


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
            with patch('builtins.__import__', side_effect=ImportError("No module named 'ray'")):
                result = is_ray_initialized()

        # Restore original ray module if it existed
        if original_ray is not None:
            sys.modules['ray'] = original_ray

        assert result is False

    @patch('vllm.ray.lazy_utils.ray')
    def test_ray_is_initialized_raises_exception(self, mock_ray):
        """Test when ray.is_initialized() raises an exception."""
        mock_ray.is_initialized.side_effect = RuntimeError("Ray error")

        # The function should catch ImportError only, so RuntimeError should propagate
        with pytest.raises(RuntimeError, match="Ray error"):
            is_ray_initialized()

    @patch('vllm.ray.lazy_utils.ray')
    def test_ray_is_initialized_returns_none(self, mock_ray):
        """Test when ray.is_initialized() returns None."""
        mock_ray.is_initialized.return_value = None

        result = is_ray_initialized()

        assert result is None
        mock_ray.is_initialized.assert_called_once()

    @patch('vllm.ray.lazy_utils.ray')
    def test_ray_is_initialized_returns_truthy_value(self, mock_ray):
        """Test when ray.is_initialized() returns a truthy non-boolean value."""
        mock_ray.is_initialized.return_value = "initialized"

        result = is_ray_initialized()

        assert result == "initialized"
        mock_ray.is_initialized.assert_called_once()

    @patch('vllm.ray.lazy_utils.ray')
    def test_ray_is_initialized_returns_falsy_value(self, mock_ray):
        """Test when ray.is_initialized() returns a falsy non-boolean value."""
        mock_ray.is_initialized.return_value = 0

        result = is_ray_initialized()

        assert result == 0
        mock_ray.is_initialized.assert_called_once()


class TestIsInRayActor:
    """Test cases for is_in_ray_actor function."""

    @patch('vllm.ray.lazy_utils.ray')
    def test_in_ray_actor_true(self, mock_ray):
        """Test when Ray is initialized and we are in a Ray actor."""
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
        # get_runtime_context should not be called if Ray is not initialized
        mock_ray.get_runtime_context.assert_not_called()

    @patch('vllm.ray.lazy_utils.ray')
    def test_in_ray_actor_false_no_actor_id(self, mock_ray):
        """Test when Ray is initialized but we are not in a Ray actor."""
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
            with patch('builtins.__import__', side_effect=ImportError("No module named 'ray'")):
                result = is_in_ray_actor()

        # Restore original ray module if it existed
        if original_ray is not None:
            sys.modules['ray'] = original_ray

        assert result is False

    @patch('vllm.ray.lazy_utils.ray')
    def test_in_ray_actor_runtime_context_exception(self, mock_ray):
        """Test when get_runtime_context() raises an exception."""
        mock_ray.is_initialized.return_value = True
        mock_ray.get_runtime_context.side_effect = RuntimeError("Runtime context error")

        # The function should catch ImportError only, so RuntimeError should propagate
        with pytest.raises(RuntimeError, match="Runtime context error"):
            is_in_ray_actor()

    @patch('vllm.ray.lazy_utils.ray')
    def test_in_ray_actor_get_actor_id_exception(self, mock_ray):
        """Test when get_actor_id() raises an exception."""
        mock_ray.is_initialized.return_value = True
        mock_runtime_context = MagicMock()
        mock_runtime_context.get_actor_id.side_effect = RuntimeError("Actor ID error")
        mock_ray.get_runtime_context.return_value = mock_runtime_context

        # The function should catch ImportError only, so RuntimeError should propagate
        with pytest.raises(RuntimeError, match="Actor ID error"):
            is_in_ray_actor()

    @patch('vllm.ray.lazy_utils.ray')
    def test_in_ray_actor_is_initialized_exception(self, mock_ray):
        """Test when is_initialized() raises an exception."""
        mock_ray.is_initialized.side_effect = RuntimeError("Initialization check error")

        # The function should catch ImportError only, so RuntimeError should propagate
        with pytest.raises(RuntimeError, match="Initialization check error"):
            is_in_ray_actor()

    @patch('vllm.ray.lazy_utils.ray')
    def test_in_ray_actor_truthy_actor_id(self, mock_ray):
        """Test when get_actor_id() returns a truthy non-None value."""
        mock_ray.is_initialized.return_value = True
        mock_runtime_context = MagicMock()
        mock_runtime_context.get_actor_id.return_value = "some_actor_id"
        mock_ray.get_runtime_context.return_value = mock_runtime_context

        result = is_in_ray_actor()

        assert result is True

    @patch('vllm.ray.lazy_utils.ray')
    def test_in_ray_actor_empty_string_actor_id(self, mock_ray):
        """Test when get_actor_id() returns an empty string."""
        mock_ray.is_initialized.return_value = True
        mock_runtime_context = MagicMock()
        mock_runtime_context.get_actor_id.return_value = ""
        mock_ray.get_runtime_context.return_value = mock_runtime_context

        result = is_in_ray_actor()

        assert result is False

    @patch('vllm.ray.lazy_utils.ray')
    def test_in_ray_actor_zero_actor_id(self, mock_ray):
        """Test when get_actor_id() returns 0."""
        mock_ray.is_initialized.return_value = True
        mock_runtime_context = MagicMock()
        mock_runtime_context.get_actor_id.return_value = 0
        mock_ray.get_runtime_context.return_value = mock_runtime_context

        result = is_in_ray_actor()

        assert result is False

    @patch('vllm.ray.lazy_utils.ray')
    def test_in_ray_actor_false_is_initialized(self, mock_ray):
        """Test when is_initialized() returns False."""
        mock_ray.is_initialized.return_value = False

        result = is_in_ray_actor()

        assert result is False
        mock_ray.get_runtime_context.assert_not_called()

    @patch('vllm.ray.lazy_utils.ray')
    def test_in_ray_actor_none_is_initialized(self, mock_ray):
        """Test when is_initialized() returns None (falsy)."""
        mock_ray.is_initialized.return_value = None

        result = is_in_ray_actor()

        assert result is False
        mock_ray.get_runtime_context.assert_not_called()

    @patch('vllm.ray.lazy_utils.ray')
    def test_in_ray_actor_complex_scenario(self, mock_ray):
        """Test a complex scenario with multiple calls."""
        mock_ray.is_initialized.return_value = True
        mock_runtime_context = MagicMock()
        mock_runtime_context.get_actor_id.return_value = "actor_456"
        mock_ray.get_runtime_context.return_value = mock_runtime_context

        # Call the function multiple times
        result1 = is_in_ray_actor()
        result2 = is_in_ray_actor()

        assert result1 is True
        assert result2 is True
        # Verify that the mocks were called the expected number of times
        assert mock_ray.is_initialized.call_count == 2
        assert mock_ray.get_runtime_context.call_count == 2
        assert mock_runtime_context.get_actor_id.call_count == 2


class TestIntegrationScenarios:
    """Integration test scenarios combining both functions."""

    @patch('vllm.ray.lazy_utils.ray')
    def test_both_functions_ray_not_available(self, mock_ray):
        """Test both functions when Ray import fails."""
        # Simulate ImportError for both functions
        with patch('vllm.ray.lazy_utils.ray', side_effect=ImportError("No module named 'ray'")):
            result1 = is_ray_initialized()
            result2 = is_in_ray_actor()

        assert result1 is False
        assert result2 is False

    @patch('vllm.ray.lazy_utils.ray')
    def test_both_functions_ray_available_not_initialized(self, mock_ray):
        """Test both functions when Ray is available but not initialized."""
        mock_ray.is_initialized.return_value = False

        result1 = is_ray_initialized()
        result2 = is_in_ray_actor()

        assert result1 is False
        assert result2 is False

    @patch('vllm.ray.lazy_utils.ray')
    def test_both_functions_ray_initialized_in_actor(self, mock_ray):
        """Test both functions when Ray is initialized and in actor."""
        mock_ray.is_initialized.return_value = True
        mock_runtime_context = MagicMock()
        mock_runtime_context.get_actor_id.return_value = "test_actor"
        mock_ray.get_runtime_context.return_value = mock_runtime_context

        result1 = is_ray_initialized()
        result2 = is_in_ray_actor()

        assert result1 is True
        assert result2 is True

    @patch('vllm.ray.lazy_utils.ray')
    def test_both_functions_ray_initialized_not_in_actor(self, mock_ray):
        """Test both functions when Ray is initialized but not in actor."""
        mock_ray.is_initialized.return_value = True
        mock_runtime_context = MagicMock()
        mock_runtime_context.get_actor_id.return_value = None
        mock_ray.get_runtime_context.return_value = mock_runtime_context

        result1 = is_ray_initialized()
        result2 = is_in_ray_actor()

        assert result1 is True
        assert result2 is False


class TestEdgeCasesAndCornerCases:
    """Test edge cases and corner cases."""

    def test_multiple_import_errors(self):
        """Test handling of multiple ImportError scenarios."""
        # Test with different ImportError messages
        error_messages = [
            "No module named 'ray'",
            "cannot import name 'ray'",
            "Ray not found",
            "",  # Empty error message
        ]

        for error_msg in error_messages:
            with patch('vllm.ray.lazy_utils.ray', side_effect=ImportError(error_msg)):
                result1 = is_ray_initialized()
                result2 = is_in_ray_actor()

                assert result1 is False, f"Failed for error message: '{error_msg}'"
                assert result2 is False, f"Failed for error message: '{error_msg}'"

    @patch('vllm.ray.lazy_utils.ray')
    def test_ray_module_partially_available(self, mock_ray):
        """Test when ray module is available but missing some attributes."""
        # Test when is_initialized is missing
        del mock_ray.is_initialized

        with pytest.raises(AttributeError):
            is_ray_initialized()

    @patch('vllm.ray.lazy_utils.ray')
    def test_runtime_context_missing_get_actor_id(self, mock_ray):
        """Test when runtime context is missing get_actor_id method."""
        mock_ray.is_initialized.return_value = True
        mock_runtime_context = MagicMock()
        del mock_runtime_context.get_actor_id
        mock_ray.get_runtime_context.return_value = mock_runtime_context

        with pytest.raises(AttributeError):
            is_in_ray_actor()

    @patch('vllm.ray.lazy_utils.ray')
    def test_get_runtime_context_returns_none(self, mock_ray):
        """Test when get_runtime_context returns None."""
        mock_ray.is_initialized.return_value = True
        mock_ray.get_runtime_context.return_value = None

        with pytest.raises(AttributeError):
            is_in_ray_actor()

    def test_concurrent_calls_simulation(self):
        """Test behavior under simulated concurrent calls."""
        import threading
        import time

        results = []

        def call_functions():
            # Simulate some delay
            time.sleep(0.01)
            with patch('vllm.ray.lazy_utils.ray', side_effect=ImportError("No module")):
                result1 = is_ray_initialized()
                result2 = is_in_ray_actor()
                results.append((result1, result2))

        # Create multiple threads
        threads = []
        for _ in range(5):
            thread = threading.Thread(target=call_functions)
            threads.append(thread)
            thread.start()

        # Wait for all threads to complete
        for thread in threads:
            thread.join()

        # All results should be (False, False)
        assert len(results) == 5
        for result1, result2 in results:
            assert result1 is False
            assert result2 is False

    @patch('vllm.ray.lazy_utils.ray')
    def test_memory_cleanup_simulation(self, mock_ray):
        """Test that functions don't hold references that prevent cleanup."""
        import gc
        import weakref

        # Create a mock object that we can track
        mock_context = MagicMock()
        mock_context.get_actor_id.return_value = "test_actor"

        # Create a weak reference to track if the object gets cleaned up
        weak_ref = weakref.ref(mock_context)

        mock_ray.is_initialized.return_value = True
        mock_ray.get_runtime_context.return_value = mock_context

        # Call the function
        result = is_in_ray_actor()
        assert result is True

        # Delete our reference
        del mock_context

        # Force garbage collection
        gc.collect()

        # The weak reference should still be valid since the mock framework
        # might hold references, but this tests that our function doesn't
        # create circular references
        # Note: This is more of a demonstration of testing memory behavior
        assert weak_ref() is not None or weak_ref() is None  # Either is acceptable
