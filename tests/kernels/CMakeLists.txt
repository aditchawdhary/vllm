# CMakeLists.txt for CUTLASS extensions tests
cmake_minimum_required(VERSION 3.18)
project(cutlass_extensions_tests LANGUAGES CXX CUDA)

# Find required packages
find_package(CUDA REQUIRED)
find_package(GTest REQUIRED)

# Set CUDA architecture
set(CMAKE_CUDA_ARCHITECTURES "75;80;86;89;90")

# Include directories
include_directories(${CMAKE_SOURCE_DIR}/csrc)
include_directories(${CMAKE_SOURCE_DIR}/csrc/cutlass_extensions)

# Add the test executable
add_executable(test_cutlass_common
    test_cutlass_common.cpp
    ${CMAKE_SOURCE_DIR}/csrc/cutlass_extensions/common.cpp
)

# Link libraries
target_link_libraries(test_cutlass_common
    ${CUDA_LIBRARIES}
    GTest::gtest_main
)
