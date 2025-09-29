# CUTLASS Common Tests

This directory contains unit tests for the `csrc/cutlass_extensions/common.cpp` file.

## Files

- `test_cutlass_common.py` - Python unit tests using pytest
- `test_cutlass_common.cpp` - C++ unit tests (optional, for direct testing)
- `CMakeLists.txt` - CMake configuration for C++ tests

## Running Python Tests

To run the Python tests:

```bash
# From the repository root
pytest tests/kernels/test_cutlass_common.py -v

# Or with CUDA-specific tests only
pytest tests/kernels/test_cutlass_common.py -v -k "cuda"
```

## Running C++ Tests (Optional)

The C++ tests are optional and can be used for more direct testing of the CUDA functions.

### Prerequisites

- CUDA Toolkit installed
- CMake 3.18 or later
- C++17 compatible compiler

### Building and Running

```bash
# From the tests/kernels directory
mkdir build
cd build
cmake ..
make
./test_cutlass_common
```

## Test Coverage

The tests cover:

- **Happy Path**: Normal operation with valid CUDA devices
- **Edge Cases**: Multiple calls, thread safety, consistency
- **Error Cases**: Non-CUDA platforms, invalid scenarios
- **Corner Cases**: Boundary conditions, format validation
