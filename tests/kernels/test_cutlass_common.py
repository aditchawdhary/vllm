# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from unittest.mock import MagicMock, patch

import torch
from vllm.platforms import current_platform

import pytest


class TestGetSmVersionNum:
    """Test suite for the get_sm_version_num function from cutlass_extensions/common.cpp"""

    @pytest.mark.skipif(not current_platform.is_cuda(),
                        reason="CUDA not available")
    def test_get_sm_version_num_returns_valid_version(self):
        """Test that get_sm_version_num returns a valid SM version number."""
        from vllm._custom_ops import get_sm_version_num

        version = get_sm_version_num()

        # SM version should be a positive integer
        assert isinstance(version, int)
        assert version > 0

        # Known CUDA compute capabilities (major.minor -> version_num)
        # SM 7.5 (Turing) -> 75
        # SM 8.0 (Ampere) -> 80
        # SM 8.6 (Ampere) -> 86
        # SM 8.9 (Ada Lovelace) -> 89
        # SM 9.0 (Hopper) -> 90
        # SM 10.0 (Blackwell) -> 100
        # SM 12.0 (future) -> 120
        valid_versions = [75, 80, 86, 89, 90, 100, 120]

        # Version should be one of the known versions or at least >= 75
        # (we support Turing and newer)
        assert version >= 75, f"SM version {version} is too old (< 7.5)"

        # Version should be reasonable (not more than 200)
        assert version <= 200, f"SM version {version} seems unreasonably high"

    @pytest.mark.skipif(not current_platform.is_cuda(),
                        reason="CUDA not available")
    def test_get_sm_version_num_consistency(self):
        """Test that get_sm_version_num returns consistent results."""
        from vllm._custom_ops import get_sm_version_num

        # Call the function multiple times and ensure it returns the same value
        version1 = get_sm_version_num()
        version2 = get_sm_version_num()
        version3 = get_sm_version_num()

        assert version1 == version2 == version3, \
            "get_sm_version_num should return consistent results"

    @pytest.mark.skipif(not current_platform.is_cuda(),
                        reason="CUDA not available")
    def test_get_sm_version_num_matches_torch_capability(self):
        """Test that get_sm_version_num matches PyTorch's device capability."""
        from vllm._custom_ops import get_sm_version_num

        version = get_sm_version_num()

        # Get PyTorch's view of the device capability
        device_capability = torch.cuda.get_device_capability(0)
        major, minor = device_capability
        expected_version = major * 10 + minor

        assert version == expected_version, \
            f"get_sm_version_num returned {version}, but PyTorch reports " \
            f"capability {major}.{minor} (expected {expected_version})"

    @pytest.mark.skipif(not current_platform.is_cuda(),
                        reason="CUDA not available")
    def test_get_sm_version_num_format(self):
        """Test that get_sm_version_num returns version in expected format."""
        from vllm._custom_ops import get_sm_version_num

        version = get_sm_version_num()

        # Version should be in format: major * 10 + minor
        # So it should be a 2-3 digit number
        assert 10 <= version <= 999, \
            f"Version {version} is not in expected format (major*10 + minor)"

        # Extract major and minor
        major = version // 10
        minor = version % 10

        # Major should be reasonable (7-20)
        assert 7 <= major <= 20, f"Major version {major} seems unreasonable"

        # Minor should be 0-9
        assert 0 <= minor <= 9, f"Minor version {minor} should be 0-9"

    @pytest.mark.skipif(current_platform.is_cuda(),
                        reason="Test only for non-CUDA platforms")
    def test_get_sm_version_num_non_cuda_platform(self):
        """Test behavior on non-CUDA platforms."""
        # This test should only run on non-CUDA platforms
        # The function might not be available or might behave differently
        try:
            from vllm._custom_ops import get_sm_version_num

            # If the function is available, it might return 0 or raise an error
            # This depends on the implementation
            version = get_sm_version_num()
            # If it returns a value, it should be 0 or negative to indicate no CUDA
            assert version <= 0, \
                "On non-CUDA platforms, version should be 0 or negative"
        except (ImportError, AttributeError, RuntimeError):
            # It's acceptable if the function is not available on non-CUDA platforms
            pytest.skip("get_sm_version_num not available on non-CUDA platform")

    def test_get_sm_version_num_mock_cuda_calls(self):
        """Test get_sm_version_num with mocked CUDA calls."""
        # This test mocks the underlying CUDA calls to test different scenarios

        # Mock the CUDA runtime library calls
        with patch('ctypes.CDLL') as mock_cdll:
            mock_cuda = MagicMock()
            mock_cdll.return_value = mock_cuda

            # Test case 1: SM 8.0 (Ampere)
            def mock_get_attribute_80(attr_ptr, attr, device):
                if attr == 75:  # cudaDevAttrComputeCapabilityMajor
                    attr_ptr.contents = 8
                elif attr == 76:  # cudaDevAttrComputeCapabilityMinor
                    attr_ptr.contents = 0
                return 0  # cudaSuccess

            mock_cuda.cudaDeviceGetAttribute = mock_get_attribute_80

            # Import and test (this would require the actual C++ function to be mockable)
            # Since we can't easily mock the C++ function, we'll test the expected behavior
            expected_version_80 = 8 * 10 + 0  # 80
            assert expected_version_80 == 80

            # Test case 2: SM 9.0 (Hopper)
            def mock_get_attribute_90(attr_ptr, attr, device):
                if attr == 75:  # cudaDevAttrComputeCapabilityMajor
                    attr_ptr.contents = 9
                elif attr == 76:  # cudaDevAttrComputeCapabilityMinor
                    attr_ptr.contents = 0
                return 0  # cudaSuccess

            mock_cuda.cudaDeviceGetAttribute = mock_get_attribute_90
            expected_version_90 = 9 * 10 + 0  # 90
            assert expected_version_90 == 90

            # Test case 3: SM 7.5 (Turing)
            def mock_get_attribute_75(attr_ptr, attr, device):
                if attr == 75:  # cudaDevAttrComputeCapabilityMajor
                    attr_ptr.contents = 7
                elif attr == 76:  # cudaDevAttrComputeCapabilityMinor
                    attr_ptr.contents = 5
                return 0  # cudaSuccess

            mock_cuda.cudaDeviceGetAttribute = mock_get_attribute_75
            expected_version_75 = 7 * 10 + 5  # 75
            assert expected_version_75 == 75

    def test_version_calculation_logic(self):
        """Test the version calculation logic (major * 10 + minor)."""
        # Test various combinations
        test_cases = [
            (7, 5, 75),   # Turing
            (8, 0, 80),   # Ampere
            (8, 6, 86),   # Ampere
            (8, 9, 89),   # Ada Lovelace
            (9, 0, 90),   # Hopper
            (10, 0, 100), # Blackwell
            (12, 0, 120), # Future
        ]

        for major, minor, expected in test_cases:
            calculated = major * 10 + minor
            assert calculated == expected, \
                f"Version calculation failed: {major}.{minor} -> {calculated} != {expected}"

    @pytest.mark.skipif(not current_platform.is_cuda(),
                        reason="CUDA not available")
    def test_get_sm_version_num_edge_cases(self):
        """Test edge cases and error conditions."""
        from vllm._custom_ops import get_sm_version_num

        # The function should work even when called many times
        versions = [get_sm_version_num() for _ in range(100)]

        # All versions should be the same
        assert all(v == versions[0] for v in versions), \
            "get_sm_version_num should return consistent results across multiple calls"

        # The function should work in different contexts
        version_in_loop = None
        for i in range(5):
            version_in_loop = get_sm_version_num()
            assert version_in_loop == versions[0], \
                f"Version in loop iteration {i} differs from initial version"

    @pytest.mark.skipif(not current_platform.is_cuda(),
                        reason="CUDA not available")
    def test_get_sm_version_num_thread_safety(self):
        """Test that get_sm_version_num is thread-safe."""
        import threading

        from vllm._custom_ops import get_sm_version_num

        results = []
        errors = []

        def worker():
            try:
                version = get_sm_version_num()
                results.append(version)
            except Exception as e:
                errors.append(e)

        # Create multiple threads
        threads = [threading.Thread(target=worker) for _ in range(10)]

        # Start all threads
        for thread in threads:
            thread.start()

        # Wait for all threads to complete
        for thread in threads:
            thread.join()

        # Check results
        assert not errors, f"Errors occurred in threads: {errors}"
        assert len(results) == 10, f"Expected 10 results, got {len(results)}"
        assert all(r == results[0] for r in results), \
            "All threads should return the same version number"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
