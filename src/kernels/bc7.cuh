// Decodes single texels of BC7-compressed textures. Kernels can use this to sample BC7 blocks from 
// anywhere, e.g. directly from a memory-mapped dds file, where hardware texture sampling is not available.
// 
// See the BC7 format reference: https://learn.microsoft.com/en-us/windows/win32/direct3d11/bc7-format-mode-reference
// The partition and anchor tables are the ones from the specification, packed into one value per partition.
//
// Also compiles without CUDA, e.g. to test against a reference decoder.

#pragma once

#include <cstdint>

#include "../types.h"

#ifdef __CUDACC__
	#define BC7_FUNC __device__ inline
	#define BC7_TABLE static __device__ const
	#define BC7_LOAD(value) __ldg(&(value))  // through the read-only cache, as threads of a warp access different entries
#else
	#define BC7_FUNC inline
	#define BC7_TABLE static const
	#define BC7_LOAD(value) (value)
#endif

// bits 0..15: subset of each texel, bits 16..19: anchor texel of subset 1
BC7_TABLE u32 BC7_PARTITIONS2[64] = {
	0x000fcccc, 0x000f8888, 0x000feeee, 0x000fecc8, 0x000fc880, 0x000ffeec, 0x000ffec8, 0x000fec80,
	0x000fc800, 0x000fffec, 0x000ffe80, 0x000fe800, 0x000fffe8, 0x000fff00, 0x000ffff0, 0x000ff000,
	0x000ff710, 0x0002008e, 0x00087100, 0x000208ce, 0x0002008c, 0x00087310, 0x00083100, 0x000f8cce,
	0x0002088c, 0x00083110, 0x00026666, 0x0002366c, 0x000817e8, 0x00080ff0, 0x0002718e, 0x0002399c,
	0x000faaaa, 0x000ff0f0, 0x00065a5a, 0x000833cc, 0x00023c3c, 0x000855aa, 0x000f9696, 0x000fa55a,
	0x000273ce, 0x000813c8, 0x0002324c, 0x00023bdc, 0x00026996, 0x000fc33c, 0x000f9966, 0x00060660,
	0x00060272, 0x000204e4, 0x00064e40, 0x00082720, 0x000fc936, 0x000f936c, 0x000239c6, 0x0002639c,
	0x000f9336, 0x000f9cc6, 0x000f817e, 0x000fe718, 0x000fccf0, 0x00020fcc, 0x00027744, 0x000fee22,
};

// 2 bits per texel: subset of each texel
BC7_TABLE u32 BC7_PARTITIONS3[64] = {
	0xaa685050, 0x6a5a5040, 0x5a5a4200, 0x5450a0a8, 0xa5a50000, 0xa0a05050, 0x5555a0a0, 0x5a5a5050,
	0xaa550000, 0xaa555500, 0xaaaa5500, 0x90909090, 0x94949494, 0xa4a4a4a4, 0xa9a59450, 0x2a0a4250,
	0xa5945040, 0x0a425054, 0xa5a5a500, 0x55a0a0a0, 0xa8a85454, 0x6a6a4040, 0xa4a45000, 0x1a1a0500,
	0x0050a4a4, 0xaaa59090, 0x14696914, 0x69691400, 0xa08585a0, 0xaa821414, 0x50a4a450, 0x6a5a0200,
	0xa9a58000, 0x5090a0a8, 0xa8a09050, 0x24242424, 0x00aa5500, 0x24924924, 0x24499224, 0x50a50a50,
	0x500aa550, 0xaaaa4444, 0x66660000, 0xa5a0a5a0, 0x50a050a0, 0x69286928, 0x44aaaa44, 0x66666600,
	0xaa444444, 0x54a854a8, 0x95809580, 0x96969600, 0xa85454a8, 0x80959580, 0xaa141414, 0x96960000,
	0xaaaa1414, 0xa05050a0, 0xa0a5a5a0, 0x96000000, 0x40804080, 0xa9a8a9a8, 0xaaaaaa44, 0x2a4a5254,
};

// bits 0..3: anchor texel of subset 1, bits 4..7: anchor texel of subset 2
BC7_TABLE u8 BC7_ANCHORS3[64] = {
	0xf3, 0x83, 0x8f, 0x3f, 0xf8, 0xf3, 0x3f, 0x8f, 0xf8, 0xf8, 0xf6, 0xf6, 0xf6, 0xf5, 0xf3, 0x83,
	0xf3, 0x83, 0xf8, 0x3f, 0xf3, 0x83, 0xf6, 0x8a, 0x35, 0xf8, 0x68, 0xa6, 0xf8, 0xf5, 0xaf, 0x8f,
	0xf8, 0x3f, 0xf3, 0xa5, 0xa6, 0x8a, 0x98, 0xaf, 0x6f, 0xf3, 0x8f, 0xf5, 0x3f, 0x6f, 0x6f, 0x8f,
	0xf3, 0x3f, 0xf5, 0xf5, 0xf5, 0xf8, 0xf5, 0xfa, 0xf5, 0xfa, 0xf8, 0xfd, 0x3f, 0xfc, 0xf3, 0x83,
};

// Parameters of each mode, 4 bits per mode, mode 0 in the lowest bits
constexpr u32 BC7_NUM_SUBSETS     = 0x21112323;
constexpr u32 BC7_PARTITION_BITS  = 0x60006664;
constexpr u32 BC7_ROTATION_BITS   = 0x00220000;
constexpr u32 BC7_SELECTION_BITS  = 0x00010000;
constexpr u32 BC7_COLOR_BITS      = 0x57757564;
constexpr u32 BC7_ALPHA_BITS      = 0x57860000;
constexpr u32 BC7_ENDPOINT_PBITS  = 0x11001001;  // one p-bit per endpoint
constexpr u32 BC7_SHARED_PBITS    = 0x00000010;  // one p-bit per subset
constexpr u32 BC7_INDEX_BITS      = 0x24222233;
constexpr u32 BC7_INDEX2_BITS     = 0x00230000;  // second set of indices of modes 4 and 5

BC7_FUNC u32 bc7Param(u32 table, u32 mode){
	return (table >> (4 * mode)) & 0xf;
}

// Interpolation weight (0 to 64) of a 2, 3 or 4 bit index
BC7_FUNC u32 bc7Weight(u32 numBits, u32 index){
	if(numBits == 2) return (0x402b1500u >> (8 * index)) & 0xff;
	if(numBits == 3) return u32(0x40372e251b120900ull >> (8 * index)) & 0xff;

	u64 weights = index < 8 ? 0x1e1a15110d090400ull : 0x403c37332f2b2622ull;
	return u32(weights >> (8 * (index & 7))) & 0xff;
}

// Reads <count> (at most 25) bits, starting at bit <offset> of the 128 bit block
BC7_FUNC u32 bc7Bits(u64 lo, u64 hi, u32 offset, u32 count){
	u64 value;
	if(offset >= 64){
		value = hi >> (offset - 64);
	}else{
		value = lo >> offset;
		if(offset + count > 64) value |= hi << (64 - offset);
	}

	return u32(value) & ((1u << count) - 1);
}

// Expands an n-bit value to 8 bits by replicating its most significant bits
BC7_FUNC u32 bc7Expand(u32 value, u32 numBits){
	value = value << (8 - numBits);

	return value | (value >> numBits);
}

BC7_FUNC u32 bc7Interpolate(u32 a, u32 b, u32 weight){
	return ((64 - weight) * a + weight * b + 32) >> 6;
}

// Decodes the texel (0 to 15, row by row) of a 16 byte BC7 block. The block only needs to be 4 byte aligned.
// Returns RGBA8, red in the lowest byte. 
BC7_FUNC u32 decodeBC7Texel(const u8* block, u32 texel){
	const u32* words = (const u32*)block;
	u64 lo = u64(words[0]) | (u64(words[1]) << 32);
	u64 hi = u64(words[2]) | (u64(words[3]) << 32);

	// the mode is given by the position of the lowest set bit
	u32 modeBits = u32(lo) & 0xff;
	if(modeBits == 0) return 0; // reserved mode, decodes to transparent black

	u32 mode = 0;
	while((modeBits & (1u << mode)) == 0) mode++;

	u32 numSubsets    = bc7Param(BC7_NUM_SUBSETS, mode);
	u32 partitionBits = bc7Param(BC7_PARTITION_BITS, mode);
	u32 rotationBits  = bc7Param(BC7_ROTATION_BITS, mode);
	u32 selectionBits = bc7Param(BC7_SELECTION_BITS, mode);
	u32 colorBits     = bc7Param(BC7_COLOR_BITS, mode);
	u32 alphaBits     = bc7Param(BC7_ALPHA_BITS, mode);
	u32 endpointPbits = bc7Param(BC7_ENDPOINT_PBITS, mode);
	u32 sharedPbits   = bc7Param(BC7_SHARED_PBITS, mode);
	u32 indexBits     = bc7Param(BC7_INDEX_BITS, mode);
	u32 index2Bits    = bc7Param(BC7_INDEX2_BITS, mode);

	u32 offset = mode + 1;
	u32 partition = bc7Bits(lo, hi, offset, partitionBits);
	offset += partitionBits;
	u32 rotation = bc7Bits(lo, hi, offset, rotationBits);
	offset += rotationBits;
	u32 selection = bc7Bits(lo, hi, offset, selectionBits);
	offset += selectionBits;

	// subset of this texel, and the anchor texels of subsets 1 and 2 (16 if there is none)
	u32 subset = 0;
	u32 anchor1 = 16;
	u32 anchor2 = 16;
	if(numSubsets == 2){
		u32 entry = BC7_LOAD(BC7_PARTITIONS2[partition]);
		subset  = (entry >> texel) & 1;
		anchor1 = (entry >> 16) & 0xf;
	}else if(numSubsets == 3){
		u32 anchors = BC7_LOAD(BC7_ANCHORS3[partition]);
		subset  = (BC7_LOAD(BC7_PARTITIONS3[partition]) >> (2 * texel)) & 3;
		anchor1 = anchors & 0xf;
		anchor2 = anchors >> 4;
	}

	// Endpoints are stored as all red values, then all green, all blue, all alpha, followed by the p-bits and the indices
	u32 numEndpoints = 2 * numSubsets;
	u32 colorOffset  = offset;
	u32 alphaOffset  = colorOffset + 3 * numEndpoints * colorBits;
	u32 pbitOffset   = alphaOffset + numEndpoints * alphaBits;
	u32 numPbits     = endpointPbits ? numEndpoints : (sharedPbits ? numSubsets : 0);
	u32 indexOffset  = pbitOffset + numPbits;

	struct Endpoint{ u32 r, g, b, a; };

	// RGBA of an endpoint, expanded to 8 bits. The p-bit, if any, is the least significant bit of each channel.
	auto readEndpoint = [&](u32 endpoint) -> Endpoint {
		u32 hasPbit = endpointPbits | sharedPbits;
		u32 pbit = 0;
		if(endpointPbits) pbit = bc7Bits(lo, hi, pbitOffset + endpoint, 1);
		if(sharedPbits)   pbit = bc7Bits(lo, hi, pbitOffset + endpoint / 2, 1);

		auto channel = [&](u32 c){
			u32 value = bc7Bits(lo, hi, colorOffset + (c * numEndpoints + endpoint) * colorBits, colorBits);
			return bc7Expand((value << hasPbit) | pbit, colorBits + hasPbit);
		};

		Endpoint result;
		result.r = channel(0);
		result.g = channel(1);
		result.b = channel(2);
		result.a = 255;

		if(alphaBits > 0){
			u32 value = bc7Bits(lo, hi, alphaOffset + endpoint * alphaBits, alphaBits);
			result.a = bc7Expand((value << hasPbit) | pbit, alphaBits + hasPbit);
		}

		return result;
	};

	Endpoint e0 = readEndpoint(2 * subset + 0);
	Endpoint e1 = readEndpoint(2 * subset + 1);

	// Indices are stored for texels 0 to 15. Anchor texels store one bit less, as their most significant bit is 0.
	u32 numAnchorsBefore = (texel > 0) + (anchor1 < texel) + (anchor2 < texel);
	u32 isAnchor = (texel == 0 || texel == anchor1 || texel == anchor2) ? 1 : 0;
	u32 index = bc7Bits(lo, hi, indexOffset + texel * indexBits - numAnchorsBefore, indexBits - isAnchor);

	u32 colorIndex = index;
	u32 colorIndexBits = indexBits;
	u32 alphaIndex = index;
	u32 alphaIndexBits = indexBits;

	if(index2Bits > 0){
		// modes 4 and 5: a second set of indices for alpha, with texel 0 as the only anchor
		u32 index2Offset = indexOffset + 16 * indexBits - 1;
		u32 index2 = bc7Bits(lo, hi, index2Offset + texel * index2Bits - (texel > 0 ? 1 : 0), index2Bits - (texel == 0 ? 1 : 0));

		alphaIndex = index2;
		alphaIndexBits = index2Bits;

		// mode 4: the selection bit swaps which set of indices is used for color and alpha
		if(selection){
			colorIndex = index2;
			colorIndexBits = index2Bits;
			alphaIndex = index;
			alphaIndexBits = indexBits;
		}
	}

	u32 colorWeight = bc7Weight(colorIndexBits, colorIndex);
	u32 alphaWeight = bc7Weight(alphaIndexBits, alphaIndex);

	u32 r = bc7Interpolate(e0.r, e1.r, colorWeight);
	u32 g = bc7Interpolate(e0.g, e1.g, colorWeight);
	u32 b = bc7Interpolate(e0.b, e1.b, colorWeight);
	u32 a = bc7Interpolate(e0.a, e1.a, alphaWeight);

	// modes 4 and 5: alpha was stored in place of one of the color channels
	u32 swap = a;
	if(rotation == 1){ a = r; r = swap; }
	if(rotation == 2){ a = g; g = swap; }
	if(rotation == 3){ a = b; b = swap; }

	return r | (g << 8) | (b << 16) | (a << 24);
}
