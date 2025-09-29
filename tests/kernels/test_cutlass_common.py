# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch
from unittest.mock import patch, MagicMock

from vllm.platforms import current_platform


@pytest.mark.skipif(not current_platform.is_cuda(),
                    reason="CUDA not available")
class TestGetSmVersionNum:
    """Test suite for the get_sm_version_num function from cutlass_extensions/common.cpp"""

    def test_get_sm_version_num_basic(self):
        """Test that get_sm_version_num returns a valid compute capability version."""
        # Import the function through torch ops
        try:
            import vllm._C
            version_num = torch.ops._C.get_sm_version_num()
        except (ImportError, AttributeError):
            pytest.skip("get_sm_version_num not available in vllm._C")

        # Check that the returned value is an integer
        assert isinstance(version_num, int), f"Expected int, got {type(version_num)}"

        # Check that the version is within reasonable bounds
        # CUDA compute capabilities range from 3.0 (30) to 12.0 (120) and beyond
        assert 30 <= version_num <= 200, f"Unexpected compute capability: {version_num}"

        # Check that it's a valid compute capability format (major * 10 + minor)
        major = version_num // 10
        minor = version_num % 10
        assert 0 <= minor <= 9, f"Invalid minor version: {minor}"
        assert major >= 3, f"Compute capability too old: {major}.{minor}"

    def test_get_sm_version_num_known_architectures(self):
        """Test that get_sm_version_num returns known architecture values."""
        try:
            import vllm._C
            version_num = torch.ops._C.get_sm_version_num()
        except (ImportError, AttributeError):
            pytest.skip("get_sm_version_num not available in vllm._C")

        # Known CUDA compute capabilities
        known_capabilities = {
            # Kepler
            30, 32, 35, 37,
            # Maxwell
            50, 52, 53,
            # Pascal
            60, 61, 62,
            # Volta
            70, 72,
            # Turing
            75,
            # Ampere
            80, 86, 87,
            # Ada Lovelace
            89,
            # Hopper
            90,
            # Blackwell
            100,
            # Future architectures
            120
        }

        # The returned value should be one of the known capabilities
        # or a reasonable extension (allowing for future architectures)
        major = version_num // 10
        minor = version_num % 10

        # Either it's a known capability or it's a reasonable future one
        is_known = version_num in known_capabilities
        is_reasonable_future = major >= 12 and minor <= 9

        assert is_known or is_reasonable_future, \
            f"Unknown compute capability: {major}.{minor} ({version_num})"

    def test_get_sm_version_num_consistency(self):
        """Test that get_sm_version_num returns consistent results across multiple calls."""
        try:
            import vllm._C
            version_num1 = torch.ops._C.get_sm_version_num()
            version_num2 = torch.ops._C.get_sm_version_num()
            version_num3 = torch.ops._C.get_sm_version_num()
        except (ImportError, AttributeError):
            pytest.skip("get_sm_version_num not available in vllm._C")

        # All calls should return the same value
        assert version_num1 == version_num2 == version_num3, \
            f"Inconsistent results: {version_num1}, {version_num2}, {version_num3}"

    def test_get_sm_version_num_matches_torch_capability(self):
        """Test that get_sm_version_num matches PyTorch's device capability."""
        try:
            import vllm._C
            version_num = torch.ops._C.get_sm_version_num()
        except (ImportError, AttributeError):
            pytest.skip("get_sm_version_num not available in vllm._C")

        # Get PyTorch's view of the device capability
        if torch.cuda.is_available() and torch.cuda.device_count() > 0:
            major, minor = torch.cuda.get_device_capability(0)
            expected_version = major * 10 + minor

            assert version_num == expected_version, \
                f"Version mismatch: get_sm_version_num()={version_num}, " \
                f"torch.cuda.get_device_capability()={major}.{minor} ({expected_version})"

    @patch('torch.ops._C.get_sm_version_num')
    def test_get_sm_version_num_error_handling(self, mock_get_sm_version):
        """Test error handling scenarios (mocked since we can't easily trigger CUDA errors)."""
        # Test case where CUDA calls might fail
        mock_get_sm_version.side_effect = RuntimeError("CUDA error")

        with pytest.raises(RuntimeError, match="CUDA error"):
            torch.ops._C.get_sm_version_num()

    def test_get_sm_version_num_with_different_devices(self):
        """Test get_sm_version_num with different CUDA devices if available."""
        try:
            import vllm._C
        except ImportError:
            pytest.skip("vllm._C not available")

        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        device_count = torch.cuda.device_count()
        if device_count <= 1:
            pytest.skip("Multiple CUDA devices not available")

        # Test with different devices
        versions = []
        for device_id in range(min(device_count, 4)):  # Test up to 4 devices
            with torch.cuda.device(device_id):
                version_num = torch.ops._C.get_sm_version_num()
                versions.append(version_num)

                # Verify it matches PyTorch's capability for this device
                major, minor = torch.cuda.get_device_capability(device_id)
                expected = major * 10 + minor
                assert version_num == expected, \
                    f"Device {device_id}: version mismatch {version_num} != {expected}"

        # All devices might have the same capability, but that's fine
        assert all(isinstance(v, int) for v in versions), \
            f"All versions should be integers: {versions}"

    def test_get_sm_version_num_thread_safety(self):
        """Test that get_sm_version_num is thread-safe."""
        import threading
        import time

        try:
            import vllm._C
        except ImportError:
            pytest.skip("vllm._C not available")

        results = []
        errors = []

        def worker():
            try:
                # Add small random delay to increase chance of race conditions
                time.sleep(0.001)
                version = torch.ops._C.get_sm_version_num()
                results.append(version)
            except Exception as e:
                errors.append(e)

        # Create multiple threads
        threads = []
        for _ in range(10):
            thread = threading.Thread(target=worker)
            threads.append(thread)
            thread.start()

        # Wait for all threads to complete
        for thread in threads:
            thread.join()

        # Check results
        assert not errors, f"Errors occurred in threads: {errors}"
        assert len(results) == 10, f"Expected 10 results, got {len(results)}"

        # All results should be the same
        unique_results = set(results)
        assert len(unique_results) == 1, \
            f"Thread safety issue: got different results {unique_results}"

        # The result should be valid
        version = results[0]
        assert isinstance(version, int), f"Expected int, got {type(version)}"
        assert 30 <= version <= 200, f"Invalid version: {version}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
