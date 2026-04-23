#pragma once


// hash map for 64bit elements comprising a 32bit key and a 32bit value
struct HashMap{

	uint64_t* entries;
	uint64_t capacity;

	// In the context of caching texture MCUs, 
	// we don't expect to encounter the max theoretically possible tex index inside the key.
	constexpr static uint32_t EMPTYKEY = 0xffffffff;
	constexpr static uint64_t EMPTYENTRY = 0xffffffff'ffffffff;

	// How often we probe for an empty slot before giving up
	constexpr static int MAX_ATTEMPTS = 10;

	// only define/compile functions for device, ignore for host
	#if defined(__CUDA_ARCH__ )

	// Murmur originates from here: https://github.com/aappleby/smhasher
	// 
	// This particular function was kindly provided by ChatGPT, 
	// but it seems to differ quite a bit in the choice of magic numbers,
	// so might have to modify if issues arise.
	// It appears this is more of a "finalizer" and not the entire hash function
	//
	// See: https://github.com/aappleby/smhasher/blob/0ff96f7835817a27d0487325b6c16033e2992eb5/src/MurmurHash3.cpp#L68
	// And: https://github.com/aappleby/smhasher/blob/0ff96f7835817a27d0487325b6c16033e2992eb5/src/MurmurHash3.cpp#L94
	// 
	static uint32_t hash_murmur(uint32_t x) {

		x ^= x >> 16;
		x *= 0x7feb352d;
		x ^= x >> 15;
		x *= 0x846ca68b;
		x ^= x >> 16;

		return x;
	}

	bool set(uint32_t key, uint32_t value, int* location, bool* alreadyExists){

		uint64_t element = (uint64_t(key) << 32) | uint64_t(value);
		uint32_t hash = hash_murmur(key);
		uint32_t hashIndex = hash % capacity;

		*alreadyExists = false;
		*location = -1;

		for(int i = 0; i < MAX_ATTEMPTS; i++){
			int probeIndex = (hashIndex + i) % capacity;
			uint64_t old = atomicCAS(&entries[probeIndex], EMPTYENTRY, element);

			if(old == EMPTYENTRY){
				*location = probeIndex;
				return true;
			}else if((old >> 32) == key){
				*location = probeIndex;
				*alreadyExists = true;
				return true;
			}
		}

		// ERROR: could not find an empty slot
		return false;
	}

	// We don't have an "empty" value, therefore the return value 
	// indicates whether the queried value exists in the hash map.
	bool get(uint32_t searchKey, uint32_t* value){

		uint32_t hash = hash_murmur(searchKey);
		uint32_t hashIndex = hash % capacity;

		for(int i = 0; i < MAX_ATTEMPTS; i++){

			int probeIndex = hashIndex + i;
			uint64_t element = entries[probeIndex];
			uint32_t elementKey = element >> 32;
			
			if(searchKey == elementKey){
				*value = element & 0xffffffff;
				return true;
			}else if(elementKey == EMPTYKEY){
				return false;
			}

			// printf("thread %d failed reading key %llu into index %d on attempt %i \n",
			// 	grid.thread_rank(), element, probeIndex, i);
		}

		return false;
	}

	bool get(uint32_t searchKey, uint32_t* value, int *location){

		uint32_t hash = hash_murmur(searchKey);
		uint32_t hashIndex = hash % capacity;

		for(int i = 0; i < MAX_ATTEMPTS; i++){

			int probeIndex = hashIndex + i;
			uint64_t element = entries[probeIndex];
			uint32_t elementKey = element >> 32;
			
			if(searchKey == elementKey){
				*value = element & 0xffffffff;
				*location = probeIndex;
				return true;
			}else if(elementKey == EMPTYKEY){
				return false;
			}
		}

		return false;
	}

	#endif

};