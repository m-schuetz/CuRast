#pragma once

#include <stdio.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <stdexcept>
#include <cmath>
#include <map>
#include <bitset>
#include "BitReader.h"
#include <string>
#include <cstring>

#define PRINT_DEBUG false

// ---------------------------------------------------------------------------
// Fast canonical Huffman decoder — no string allocation, no map lookup.
// Accumulates bits and compares against min/max code ranges per bit-length,
// giving O(code_length) integer operations instead of O(L * log N) string ops.
// ---------------------------------------------------------------------------
struct FastHuffmanTable {
    int     mincode[16] = {};
    int     maxcode[16] = {};   // -1 means no codes of this bit-length
    int     valptr[16]  = {};
    uint8_t huffval[256] = {};

    int decode(BitReader& reader) const {
        int code = 0;
        for (int i = 0; i < 16; i++) {
            code = (code << 1) | reader.read_bit();
            if (maxcode[i] >= 0 && code <= maxcode[i])
                return huffval[valptr[i] + code - mincode[i]];
        }
        throw std::runtime_error("Invalid Huffman code");
    }
};

class JPEGIndexer {
private:
    struct Component {
        int h_sampling;
        int v_sampling;
    };
    std::string jpeg_path;
    int compressed_data_start = 0;
    std::vector<Component> components;

    std::vector<uint8_t> encoded_stream;

    std::vector<int> starts;
    std::vector<int> ranges;
    int only_ac_bit_count = 0;

    // Fast canonical tables: [class 0=DC/1=AC][table_id 0..3]
    FastHuffmanTable fast_tables[2][4];
    // Per-component pointers into fast_tables, set during SOS parsing.
    const FastHuffmanTable* fast_huff_comp[2][4] = {};

    // ------------------------------------------------------------------
    // Append num_bits bits from src (starting at start_bit) into only_ac_data.
    // Processes 8 bits per iteration — ~8x faster than 1-bit-at-a-time.
    // ------------------------------------------------------------------
    void append_ac_bits(const std::vector<uint8_t>& src, int start_bit, int num_bits) {
        if (num_bits == 0) return;

        int needed_bytes = (only_ac_bit_count + num_bits + 7) / 8;
        if (needed_bytes > (int)only_ac_data.size())
            only_ac_data.resize(needed_bytes, 0);

        const uint8_t* s  = src.data();
        int            sz = (int)src.size();
        uint8_t*       d  = only_ac_data.data();

        int src_bit   = start_bit;
        int dst_bit   = only_ac_bit_count;
        int remaining = num_bits;

        while (remaining > 0) {
            int chunk    = remaining < 8 ? remaining : 8;
            int src_byte = src_bit >> 3;
            int src_off  = src_bit & 7;

            // Read up to 16 bits from source to handle any byte-boundary alignment.
            uint16_t src_word = ((uint16_t)s[src_byte] << 8) |
                                (src_byte + 1 < sz ? (uint16_t)s[src_byte + 1] : 0u);
            // Extract 'chunk' bits starting at src_off (MSB-first).
            uint8_t byte_val = (uint8_t)((src_word >> (16 - src_off - chunk)) &
                                          ((1u << chunk) - 1u));

            int dst_byte = dst_bit >> 3;
            int dst_off  = dst_bit & 7;
            if (dst_off + chunk <= 8) {
                d[dst_byte] |= byte_val << (8 - dst_off - chunk);
            } else {
                int first  = 8 - dst_off;
                int second = chunk - first;
                d[dst_byte]     |= byte_val >> second;
                d[dst_byte + 1] |= (uint8_t)(byte_val << (8 - second));
            }

            src_bit   += chunk;
            dst_bit   += chunk;
            remaining -= chunk;
        }

        only_ac_bit_count += num_bits;
    }

    // Write 'count' bits of 'val' (MSB-first) into only_ac_data.
    void write_bits_to_ac(uint32_t val, int count) {
        int needed_bytes = (only_ac_bit_count + count + 7) / 8;
        if (needed_bytes > (int)only_ac_data.size())
            only_ac_data.resize(needed_bytes, 0);

        uint8_t* d         = only_ac_data.data();
        int      dst_bit   = only_ac_bit_count;
        int      remaining = count;

        while (remaining > 0) {
            int dst_byte = dst_bit >> 3;
            int dst_off  = dst_bit & 7;
            int chunk    = 8 - dst_off;
            if (chunk > remaining) chunk = remaining;
            uint8_t bits = (uint8_t)((val >> (remaining - chunk)) & ((1u << chunk) - 1u));
            d[dst_byte] |= bits << (8 - dst_off - chunk);
            dst_bit   += chunk;
            remaining -= chunk;
        }

        only_ac_bit_count += count;
    }

    void parse_jpeg_headers(std::vector<uint8_t> data) {
        int index = 0;
        int length = 0;
        int precision = 0;
        while (index < (int)data.size()) {
            uint16_t marker = (data[index] << 8) | data[index + 1];
            index += 2;

            switch (marker) {
            case 0xFFD8:
                if (PRINT_DEBUG) std::cout << "Start of Image (SOI) found\n";
                break;
            case 0xFFC0:
                if (PRINT_DEBUG) std::cout << "Start of Frame (SOF0)\n";
                length = (data[index] << 8) | data[index + 1];
                index += 2;
                precision = data[index++];
                height = (data[index] << 8) | data[index + 1];
                index += 2;
                width = (data[index] << 8) | data[index + 1];
                index += 2;
                color_components = data[index++];

                if (PRINT_DEBUG) std::cout << "Image size: " << width << "x" << height << "\n";
                if (PRINT_DEBUG) std::cout << "Number of color components: " << color_components << "\n";
                components.clear();
                for (int i = 0; i < color_components; ++i) {
                    int component_id     = data[index++];
                    int sampling_factors = data[index++];
                    int quant_table_id   = data[index++];

                    int h_sampling = (sampling_factors >> 4) & 0x0F;
                    int v_sampling = sampling_factors & 0x0F;
                    components.push_back({ h_sampling, v_sampling });
                    if (PRINT_DEBUG) std::cout << "Component ID: " << component_id
                        << ", h_sampling: " << h_sampling
                        << ", v_sampling: " << v_sampling
                        << ", Quantization Table ID: " << quant_table_id << "\n";
                }

                index += (length - (8 + color_components * 3));
                break;
            case 0xFFC4: // DHT (Define Huffman Table)
                if (PRINT_DEBUG) std::cout << "Define Huffman Table\n";
                parse_huffman_table(data, index);
                break;
            case 0xFFDB: // DQT (Define Quantization Table)
                if (PRINT_DEBUG) std::cout << "Define Quantization Table\n";
                parse_quantization_table(data, index);
                break;
            case 0xFFDA: // SOS (Start of Scan)
            {
                length = (data[index] << 8) | data[index + 1];
                int num_components_in_scan = data[index + 2];
                std::vector<uint8_t> sos_data(data.begin() + index + 2, data.begin() + index + length);

                for (int i = 0; i < num_components_in_scan; i++) {
                    int component_id             = sos_data[1 + 2 * i];
                    int huffman_table_assignment = sos_data[2 + 2 * i];
                    int dc_huff_table_id = (huffman_table_assignment >> 4) & 0x0F;
                    int ac_huff_table_id = huffman_table_assignment & 0x0F;
                    huffman_tables_components[0][i] = huffman_tables[0][dc_huff_table_id];
                    huffman_tables_components[1][i] = huffman_tables[1][ac_huff_table_id];

                    // Wire up fast tables for the hot decode path.
                    fast_huff_comp[0][i] = &fast_tables[0][dc_huff_table_id];
                    fast_huff_comp[1][i] = &fast_tables[1][ac_huff_table_id];

                    if (PRINT_DEBUG) std::cout << "Component ID: " << component_id
                        << ", DC Table: " << dc_huff_table_id
                        << ", AC Table: " << ac_huff_table_id << "\n";
                }
                index += length;
                compressed_data_start = index;
                if (PRINT_DEBUG) std::cout << "Start of Scan (SOS) found\n";
                if (PRINT_DEBUG) std::cout << "Number of components in scan: " << num_components_in_scan << "\n";
                return;
            }
            }
        }
    }

    std::map<std::string, int> build_huffman_table(int num_codes_per_bit_length[16], const std::vector<int>& huffman_values) {
        std::map<std::string, int> huffman_table;
        int code = 0;
        int value_index = 0;
        for (int bit_length = 1; bit_length <= 16; ++bit_length) {
            int num_codes = num_codes_per_bit_length[bit_length - 1];
            for (int i = 0; i < num_codes; ++i) {
                std::string binary_code = std::bitset<16>(code).to_string().substr(16 - bit_length);
                huffman_table[binary_code] = huffman_values[value_index++];
                code++;
            }
            code <<= 1;
        }
        return huffman_table;
    }

    void parse_huffman_table(const std::vector<uint8_t>& data, int& index) {
        int length = (data[index] << 8) | data[index + 1];
        int end    = index + length;
        index += 2;
        while (index < end) {
            uint8_t table_info  = data[index++];
            int     table_class = (table_info >> 4) & 0x0F;
            int     table_id    = table_info & 0x0F;
            int num_codes_per_bit_length[16];
            for (int i = 0; i < 16; ++i)
                num_codes_per_bit_length[i] = data[index++];
            int total_codes = 0;
            for (int i = 0; i < 16; ++i) total_codes += num_codes_per_bit_length[i];
            std::vector<int> huffman_values(data.begin() + index, data.begin() + index + total_codes);
            index += total_codes;

            // Build legacy string-map (used by LargeGlbLoader to upload to GPU).
            huffman_tables[table_class][table_id] =
                build_huffman_table(num_codes_per_bit_length, huffman_values);

            // Build fast canonical table used in the hot decode loop.
            FastHuffmanTable& ft = fast_tables[table_class][table_id];
            int code = 0, vi_src = 0;
            for (int i = 0; i < 16; i++) {
                int n = num_codes_per_bit_length[i];
                if (n == 0) {
                    ft.maxcode[i] = -1;
                    ft.mincode[i] = 0;
                    ft.valptr[i]  = vi_src;
                } else {
                    ft.mincode[i] = code;
                    ft.valptr[i]  = vi_src;
                    for (int j = 0; j < n; j++)
                        ft.huffval[vi_src + j] = (uint8_t)huffman_values[vi_src + j];
                    vi_src += n;
                    code   += n;
                    ft.maxcode[i] = code - 1;
                }
                code <<= 1;
            }

            if (PRINT_DEBUG)
                std::cout << (table_class == 0 ? "DC" : "AC") << " Huffman Table (ID=" << table_id << ") parsed.\n";
        }
    }

    void parse_quantization_table(const std::vector<uint8_t>& data, int& index) {
        int length = (data[index] << 8) | data[index + 1];
        int end    = index + length;
        index += 2;

        while (index < end) {
            uint8_t table_info = data[index++];
            int precision  = table_info >> 4;
            int table_id   = table_info & 0x0F;
            int table_size = (precision == 0) ? 64 : 128;
            std::vector<int> quant_table;
            if (precision == 0) {
                quant_table.assign(data.begin() + index, data.begin() + index + 64);
            } else {
                quant_table.reserve(64);
                for (int i = 0; i < 64; ++i) {
                    int val = (data[index + i * 2] << 8) | data[index + i * 2 + 1];
                    quant_table.push_back(val);
                }
            }
            quantization_tables[table_id] = quant_table;
            index += table_size;
        }
    }

    std::vector<uint8_t> RemoveFF00(const std::vector<uint8_t>& data) {
        std::vector<uint8_t> cleaned_data;
        cleaned_data.reserve(data.size());
        size_t length = data.size();
        for (size_t i = 0; i < length; ++i) {
            if (data[i] == 0xFF) {
                if (i + 1 < length && data[i + 1] == 0x00) {
                    cleaned_data.push_back(data[i]);
                    i++;
                } else if (i + 1 < length && data[i + 1] == 0xD9) {
                    break;
                }
            } else {
                cleaned_data.push_back(data[i]);
            }
        }
        return cleaned_data;
    }

    int extend_value(int value, int size) {
        if (size == 0) return 0;
        int mask = (1 << (size - 1));
        if (value < mask)
            value -= (1 << size) - 1;
        return value;
    }

    int decode_dc(int component_id, BitReader& bit_reader) {
        int huffman_code = fast_huff_comp[0][component_id]->decode(bit_reader);
        if (huffman_code > 0) {
            int dc_difference = bit_reader.read_bits(huffman_code);
            return extend_value(dc_difference, huffman_code);
        }
        return 0;
    }

    void decode_ac(int component_id, BitReader& bit_reader) {
        const FastHuffmanTable* ft = fast_huff_comp[1][component_id];
        int ac_start_bit = bit_reader.get_bit_position();
        int idx = 0;
        while (idx < 63) {
            int huffman_code = ft->decode(bit_reader);
            if (huffman_code == 0x00)
                break;
            int size = huffman_code;
            if (huffman_code > 15) {
                int run_length = huffman_code >> 4;
                size = huffman_code & 0x0F;
                if (run_length + idx > 63)
                    break;
                idx += run_length;
            }
            bit_reader.read_bits(size);
            idx++;
        }

        int ac_end_bit = bit_reader.get_bit_position();
        starts.push_back(ac_start_bit);
        ranges.push_back(ac_end_bit - ac_start_bit);
    }

    void build_mcu_index(std::vector<uint8_t> compressed_data) {
        encoded_stream    = RemoveFF00(compressed_data);
        only_ac_bit_count = 0;
        only_ac_data.reserve(encoded_stream.size());

        BitReader bit_reader(encoded_stream);
        for (const auto& comp : components)
            previous_dc_values.push_back(0);

        int idx = 0;
        uint32_t packed_offset         = 0;
        int relative_offsets_count     = 0;
        int largest_relative_offset    = 0;
        uint32_t last_absolute_offset  = 0;
        int offset_counter             = 0;

        while (!bit_reader.is_at_end()) {
            for (size_t comp_idx = 0; comp_idx < components.size(); ++comp_idx) {
                const auto& comp = components[comp_idx];
                for (int vs = 0; vs < comp.v_sampling; ++vs) {
                    for (int hs = 0; hs < comp.h_sampling; ++hs) {
                        try {
                            if (bit_reader.is_at_end())
                                return;
                            int dc_coefficient = 0;

                            if (idx % 6 == 0) {
                                absoluteMcuOffsets.push_back(only_ac_bit_count);
                                uint32_t current_ac_offset = only_ac_bit_count;
                                if (offset_counter % 9 == 0) {
                                    last_absolute_offset = current_ac_offset;
                                    packed_offset        = last_absolute_offset;
                                    mcu_index.push_back(packed_offset);
                                    packed_offset          = 0;
                                    relative_offsets_count = 0;
                                } else {
                                    uint32_t relative_offset = current_ac_offset - last_absolute_offset;
                                    if ((int)relative_offset > largest_relative_offset)
                                        largest_relative_offset = (int)relative_offset;
                                    packed_offset |= ((relative_offset & 0xFFFF) << (16 * (1 - relative_offsets_count)));
                                    relative_offsets_count++;
                                    if (relative_offsets_count == 2) {
                                        mcu_index.push_back(packed_offset);
                                        packed_offset          = 0;
                                        relative_offsets_count = 0;
                                    }
                                }
                                offset_counter++;
                            }

                            if (idx % 6 == 0 || idx % 6 == 4 || idx % 6 == 5) {
                                int dc_start_bit = bit_reader.get_bit_position();
                                dc_coefficient   = previous_dc_values[comp_idx] + decode_dc(comp_idx, bit_reader);
                                int dc_numBits   = bit_reader.get_bit_position() - dc_start_bit;

                                bitsPerDUsDC[idx % 6] += dc_numBits;
                                DUsDCCounter[idx % 6] += 1;

                                // Write a fixed 12-bit DC value into the AC data stream.
                                uint16_t lsb_12 = (uint16_t)((dc_coefficient + 2048) & 0x0FFF);
                                write_bits_to_ac(lsb_12, 12);
                            } else {
                                int dc_start_bit = bit_reader.get_bit_position();
                                dc_coefficient   = previous_dc_values[comp_idx] + decode_dc(comp_idx, bit_reader);
                                int dc_end_bit   = bit_reader.get_bit_position();
                                int dc_numBits   = dc_end_bit - dc_start_bit;

                                bitsPerDUsDC[idx % 6] += dc_numBits;
                                DUsDCCounter[idx % 6] += 1;

                                append_ac_bits(encoded_stream, dc_start_bit, dc_numBits);
                            }
                            previous_dc_values[comp_idx] = dc_coefficient;
                            decode_ac(comp_idx, bit_reader);
                            if (idx % 6 == 5) {
                                for (int i = 0; i < (int)starts.size(); i++)
                                    append_ac_bits(encoded_stream, starts[i], ranges[i]);
                                starts.clear();
                                ranges.clear();
                            }
                        } catch (const std::exception& e) {
                            if (PRINT_DEBUG) std::cerr << "Decoding error at MCU " << mcu_index.size()
                                << " and bit position " << bit_reader.get_bit_position()
                                << ": " << e.what() << std::endl;
                            if (PRINT_DEBUG) std::cout << "Largest relative offset is: " << largest_relative_offset << "\n";
                            if (PRINT_DEBUG) std::cout << "JPEG bytes: " << image_data.size() << "\n";
                            return;
                        }
                        idx++;
                    }
                }
            }
        }
        if (packed_offset != 0)
            mcu_index.push_back(packed_offset);

        if (PRINT_DEBUG) std::cout << "Largest relative offset is: " << largest_relative_offset << "\n";
        if (PRINT_DEBUG) std::cout << "JPEG bytes: " << image_data.size() * 8 << "\n";
    }

public:

    std::vector<uint8_t> image_data;
    std::map<int, std::map<int, std::map<std::string, int>>> huffman_tables;
    std::map<int, std::map<int, std::map<std::string, int>>> huffman_tables_components;
    std::map<int, std::vector<int>> quantization_tables;
    std::vector<int> previous_dc_values;
    std::vector<uint32_t> mcu_index;
    std::vector<uint8_t> only_ac_data;

    JPEGIndexer() {}

    JPEGIndexer(std::vector<uint8_t>& data) {
        parse_jpeg_headers(data);
        this->image_data = std::vector<uint8_t>(data.begin() + compressed_data_start, data.end());
        build_mcu_index(this->image_data);
    }

    int width = 0;
    int height = 0;
    int color_components = 0;

    int bitsPerDUsDC[6] = {0};
    int DUsDCCounter[6] = {0};
    int mipMapLevel = 0;
    std::vector<uint32_t> absoluteMcuOffsets;
};
