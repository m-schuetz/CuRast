#pragma once
#include <vector>
#include <stdexcept>
#include <cstdint>

class BitReader {
private:
    const uint8_t* data_ptr;
    int data_size;
    uint64_t cache = 0;
    int cache_bits = 0;
    int byte_pos = 0;
    int end_bit_position;

    void fill_cache() {
        while (cache_bits <= 56 && byte_pos < data_size) {
            cache = (cache << 8) | data_ptr[byte_pos++];
            cache_bits += 8;
        }
    }

public:
    BitReader(const std::vector<uint8_t>& data)
        : data_ptr(data.data()), data_size((int)data.size()),
          end_bit_position((int)data.size() * 8) {
        fill_cache();
    }

    BitReader(std::vector<uint8_t>& data, int start_bit, int end_bit)
        : data_ptr(data.data()), data_size((int)data.size()),
          end_bit_position(end_bit) {
        byte_pos = start_bit / 8;
        fill_cache();
        // Discard the leading bits before start_bit within the first byte
        cache_bits -= start_bit % 8;
    }

    bool is_at_end() const {
        return get_bit_position() >= end_bit_position;
    }

    int read_bit() {
        if (cache_bits == 0) {
            fill_cache();
            if (cache_bits == 0) throw std::runtime_error("End of bitstream reached");
        }
        --cache_bits;
        return (int)((cache >> cache_bits) & 1ULL);
    }

    // Read up to 32 bits at once — much faster than calling read_bit() repeatedly.
    int read_bits(int count) {
        if (count == 0) return 0;
        while (cache_bits < count && byte_pos < data_size) {
            cache = (cache << 8) | data_ptr[byte_pos++];
            cache_bits += 8;
        }
        cache_bits -= count;
        return (int)((cache >> cache_bits) & ((1ULL << count) - 1));
    }

    int get_bit_position() const {
        return byte_pos * 8 - cache_bits;
    }

    void jump_to_position(int bit_position) {
        byte_pos = bit_position / 8;
        cache = 0;
        cache_bits = 0;
        fill_cache();
        // Discard leading bits before bit_position within that byte
        cache_bits -= bit_position % 8;
    }
};
