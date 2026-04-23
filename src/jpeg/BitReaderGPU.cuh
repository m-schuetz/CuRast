#pragma once
class BitReaderGPU {
private:
    const uint8_t* data;
    int byte_offset = 0;
    int bit_offset = 0;

public:


    __device__ BitReaderGPU(const uint8_t* data, int start_bit_position)
        : data(data) {
        bit_offset = start_bit_position;
        }
    
    __device__ int read_bit() {
        int bit = (data[byte_offset] >> (7 - bit_offset)) & 1;
        if (++bit_offset == 8) {
            bit_offset = 0;
            byte_offset++;
        }
        return bit;
    }

	uint32_t advance(uint32_t numBits){

		bit_offset += numBits;
		byte_offset += bit_offset / 8;
		bit_offset = bit_offset % 8;
	}

	uint32_t peek16Bit(){

		int _byte_offset = byte_offset;
		int _bit_offset = bit_offset;

		uint32_t value = 0;

		for(int i = 0; i < 16; i++){
			int bit = (data[_byte_offset] >> (7 - _bit_offset)) & 1;
			if (++_bit_offset == 8) {
				_bit_offset = 0;
				_byte_offset++;
			}

			value = (value << 1) | bit;
		}
		
		return value;
	}

	uint32_t peek16Bit2(){

		uint32_t value = 0;

		value = value | ((data[byte_offset + 0] << 24));
		value = value | ((data[byte_offset + 1] << 16));
		value = value | ((data[byte_offset + 2] << 8));

		value = value >> (16 - bit_offset);
		value = value & 0xffff;

		return value;
	}

	int read_bits(int count) {

		// if(count > 16) printf("damn...\n");

		uint32_t peeked = peek16Bit2();
		uint32_t value = peeked >> (16 - count);

		advance(count);

		// int value = 0;
		// for (int i = 0; i < count; i++) {
		// 	value = (value << 1) | read_bit();
		// }
		return value;
	}

    __device__ int get_bit_position() const {
        return byte_offset * 8 + bit_offset;
    }

	static void printBinary(uint32_t n){

		for (int i = 0; i < 32; i++) {

			uint32_t bit = (n >> i) & 1;

			if(bit){
				printf("1");
			}else{
				printf("0");
			}
			// printf("%c", (n & (1U << i)) ? '1' : '0');
			if (i % 8 == 0 && i != 0) {
				printf(" ");  // Optional space every 8 bits
			}
		}
		printf("\n");
	}
};


//##################################################
//
// A small test program to experiment/verify bit reading patterns
// - peek16Bit_jpeg and peek16Bit_jpeged don't match with the bitpattern because they reverse bits in each byte (little vs big endiann-ish)
//
//##################################################

// #include <iostream>
// #include <print>
// #include <format>
// #include <stdint.h>
// #include <cstring>

// using namespace std;

// uint32_t peek16Bit(uint8_t* data, uint64_t bitOffset){
    
//     uint32_t value = 0;
//     uint64_t byteOffset = bitOffset / 8llu;
    
//     value = value | (data[byteOffset + 0] << 0);
//     value = value | (data[byteOffset + 1] << 8);
//     value = value | (data[byteOffset + 2] << 16);
    
//     value = value >> (bitOffset % 8);
//     value = value & 0xffffff;
    
//     return value;
// }

// uint32_t peek16Bit_jpeg(uint8_t* data, uint64_t byte_offset, uint64_t bit_offset){

// 	int _byte_offset = byte_offset;
// 	int _bit_offset = bit_offset;

// 	uint32_t value = 0;

// 	for(int i = 0; i < 16; i++){
// 		int bit = (data[_byte_offset] >> (7 - _bit_offset)) & 1;
// 		if (++_bit_offset == 8) {
// 			_bit_offset = 0;
// 			_byte_offset++;
// 		}
		
// 		value = (value << 1) | bit;
// 	}
	
// 	return value;
// }

// uint32_t peek16Bit_jpeged(uint8_t* data, uint64_t byte_offset, uint64_t bit_offset){
    
//     uint32_t value = 0;
    
//     value = value | ((data[byte_offset + 0] << 24));
//     value = value | ((data[byte_offset + 1] << 16));
//     value = value | ((data[byte_offset + 2] << 8));
    
//     value = value >> (16 - bit_offset);
//     value = value & 0xffff;
    
//     return value;
// }

// void printBinaryString(uint8_t* data){
//     string str = "";
//     for(int i = 0; i < 16; i++){
        
//         string tmp = format("{:08b}", data[i]);
//         reverse(tmp.begin(), tmp.end());
        
//         str += tmp;
//         //str += "'";
//     }
    
//     println("     {}", str);
// }

// int main() {
//     uint8_t data[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25};
//     uint32_t* u32 = (uint32_t*)data;
    
//     for(int i = 0; i < 3; i++){
//         std::string indent(i, ' ');
    
//         if(i % 8 == 0){
//             printBinaryString(data);
//         }
        
//         uint16_t bits = peek16Bit(data, i);
//         string formatted = format("{:016b}", bits);
//         reverse(formatted.begin(), formatted.end());
//         println("{:3d}: {}{}", i, indent, formatted);
//     }
    
//     for(int i = 0; i < 16; i++){
//         std::string indent(i, ' ');
    
//         if(i % 8 == 0){
//             printBinaryString(data);
//         }
        
//         int64_t byteOffset = i / 8;
//         int64_t bitOffset = i % 8;
        
//         uint16_t bits = peek16Bit_jpeg(data, byteOffset, bitOffset);
//         string formatted = format("{:016b}", bits);
//         reverse(formatted.begin(), formatted.end());
//         println("{:3d}: {}{}", i, indent, formatted);
        
//         uint16_t bits_peged = peek16Bit_jpeged(data, byteOffset, bitOffset);
//         formatted = format("{:016b}", bits_peged);
//         reverse(formatted.begin(), formatted.end());
//         println("{:3d}: {}{}", i, indent, formatted);
//     }
    

//     return 0;
// }