#include "batch_invariant.hpp"
#include "../ops.h"

// Wrapper function to expose the C++ function to Python
bool vllm_kernel_override_batch_invariant_cpp() {
  return vllm::vllm_kernel_override_batch_invariant();
}
