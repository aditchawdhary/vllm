// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <climits>
#include "cutlass_extensions/common.hpp"

class GetSmVersionNumTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Check if CUDA is available
        int device_count;
        cudaError_t error = cudaGetDeviceCount(&device_count);
        if (error != cudaSuccess || device_count == 0) {
            GTEST_SKIP() << "CUDA not available or no CUDA devices found";
        }
    }
};

TEST_F(GetSmVersionNumTest, ReturnsValidVersion) {
    int32_t version = get_sm_version_num();

    // Version should be positive
    EXPECT_GT(version, 0) << "SM version should be positive";

    // Version should be reasonable (>= 75 for Turing and newer)
    EXPECT_GE(version, 75) << "SM version should be at least 75 (Turing)";

    // Version should not be unreasonably high
    EXPECT_LE(version, 200) << "SM version should not exceed 200";
}

TEST_F(GetSmVersionNumTest, ConsistentResults) {
    int32_t version1 = get_sm_version_num();
    int32_t version2 = get_sm_version_num();
    int32_t version3 = get_sm_version_num();

    EXPECT_EQ(version1, version2) << "Multiple calls should return same version";
    EXPECT_EQ(version2, version3) << "Multiple calls should return same version";
}

TEST_F(GetSmVersionNumTest, MatchesTorchCapability) {
    int32_t version = get_sm_version_num();

    // Get the capability directly from CUDA
    int major, minor;
    cudaError_t error1 = cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, 0);
    cudaError_t error2 = cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, 0);

    ASSERT_EQ(error1, cudaSuccess) << "Failed to get major capability";
    ASSERT_EQ(error2, cudaSuccess) << "Failed to get minor capability";

    int32_t expected_version = major * 10 + minor;
    EXPECT_EQ(version, expected_version)
        << "get_sm_version_num returned " << version
        << " but CUDA reports " << major << "." << minor
        << " (expected " << expected_version << ")";
}

TEST_F(GetSmVersionNumTest, VersionFormat) {
    int32_t version = get_sm_version_num();

    // Version should be in format: major * 10 + minor
    EXPECT_GE(version, 10) << "Version should be at least 10 (1.0)";
    EXPECT_LE(version, 999) << "Version should not exceed 999 (99.9)";

    // Extract major and minor
    int major = version / 10;
    int minor = version % 10;

    // Major should be reasonable
    EXPECT_GE(major, 7) << "Major version should be at least 7";
    EXPECT_LE(major, 20) << "Major version should not exceed 20";

    // Minor should be 0-9
    EXPECT_GE(minor, 0) << "Minor version should be at least 0";
    EXPECT_LE(minor, 9) << "Minor version should not exceed 9";
}

TEST_F(GetSmVersionNumTest, KnownArchitectures) {
    int32_t version = get_sm_version_num();

    // Test against known CUDA architectures
    std::vector<int32_t> known_versions = {
        75,  // Turing (RTX 20xx, GTX 16xx)
        80,  // Ampere (A100)
        86,  // Ampere (RTX 30xx)
        89,  // Ada Lovelace (RTX 40xx)
        90,  // Hopper (H100)
        100, // Blackwell (B100)
        120  // Future architecture
    };

    // Version should be one of the known versions or at least >= 75
    bool is_known_or_newer = false;
    for (int32_t known_version : known_versions) {
        if (version == known_version) {
            is_known_or_newer = true;
            break;
        }
    }

    // If not a known version, it should at least be >= 75
    if (!is_known_or_newer) {
        EXPECT_GE(version, 75) << "Unknown version " << version << " should be at least 75";
    }
}

TEST_F(GetSmVersionNumTest, ErrorHandling) {
    // Test that the function doesn't crash or return invalid values
    // even if called multiple times rapidly
    for (int i = 0; i < 1000; ++i) {
        int32_t version = get_sm_version_num();
        EXPECT_GT(version, 0) << "Version should be positive on iteration " << i;
    }
}

TEST_F(GetSmVersionNumTest, ThreadSafety) {
    const int num_threads = 10;
    const int calls_per_thread = 100;
    std::vector<std::thread> threads;
    std::vector<std::vector<int32_t>> results(num_threads);

    // Launch threads
    for (int t = 0; t < num_threads; ++t) {
        threads.emplace_back([&results, t, calls_per_thread]() {
            for (int i = 0; i < calls_per_thread; ++i) {
                results[t].push_back(get_sm_version_num());
            }
        });
    }

    // Wait for all threads
    for (auto& thread : threads) {
        thread.join();
    }

    // Verify all results are consistent
    int32_t expected_version = results[0][0];
    for (int t = 0; t < num_threads; ++t) {
        for (int i = 0; i < calls_per_thread; ++i) {
            EXPECT_EQ(results[t][i], expected_version)
                << "Thread " << t << ", call " << i << " returned different version";
        }
    }
}

// Test the function behavior with different device contexts
TEST_F(GetSmVersionNumTest, MultipleDevices) {
    int device_count;
    cudaGetDeviceCount(&device_count);

    if (device_count > 1) {
        // Test that the function works correctly when switching devices
        for (int device = 0; device < device_count; ++device) {
            cudaSetDevice(device);
            int32_t version = get_sm_version_num();
            EXPECT_GT(version, 0) << "Version should be positive for device " << device;

            // Get expected version for this device
            int major, minor;
            cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device);
            cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, device);
            int32_t expected = major * 10 + minor;

            EXPECT_EQ(version, expected)
                << "Device " << device << " version mismatch";
        }
    } else {
        GTEST_SKIP() << "Multiple device test requires more than one CUDA device";
    }
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
