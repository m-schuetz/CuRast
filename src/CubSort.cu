#if defined(USE_HIP) || defined(__HIP_PLATFORM_AMD__)
// hipCUB mirrors the CUB device-wide interface, so the call sites below stay in
// the CUB spelling on both backends.
// Only the radix sort is needed. The hipcub.hpp umbrella header additionally
// pulls in device_for.hpp, which requires a C++23 <mdspan> implementation that
// libstdc++ 13 does not ship.
#include <hipcub/device/device_radix_sort.hpp>
namespace cub = hipcub;
#include "cuda_to_hip.h"
#else
#include <cub/cub.cuh>
#include <cuda_runtime.h>
#endif
#include <cstdint>

static void*  s_sortTempStorage      = nullptr;
static size_t s_sortTempStorageBytes = 0;

// Host-callable CUB radix sort for uint64_t keys (ascending).
// d_keys_in and d_keys_out must each have at least num_items elements.
void cubSortUint64Keys(uint64_t* d_keys_in, uint64_t* d_keys_out, int num_items) {
	size_t temp_bytes = 0;
	cub::DeviceRadixSort::SortKeys(nullptr, temp_bytes, d_keys_in, d_keys_out, num_items);

	if (temp_bytes > s_sortTempStorageBytes) {
		cudaFree(s_sortTempStorage);
		cudaMalloc(&s_sortTempStorage, temp_bytes);
		s_sortTempStorageBytes = temp_bytes;
	}

	cub::DeviceRadixSort::SortKeys(s_sortTempStorage, temp_bytes, d_keys_in, d_keys_out, num_items);
}
