// mesh_visbuffer.slang — Visibility buffer pass 1
// Writes the global triangle index (across all meshes) into a R32_UINT render target.
// The resolve pass uses a cumulative triangle count array to recover mesh + local triangle index.
// slangc mesh_visbuffer.slang -target spirv -fvk-use-entrypoint-name -capability spvBindlessTextureNV -source-embed-style u32 -source-embed-name mesh_visbuffer_spv -o mesh_visbuffer.h

struct Node {
	column_major float4x4 world;
	column_major float4x4 worldViewProj;
	float3 *position;  // unused — position comes from vertex buffer
	float2 *uv;        // unused — uv comes from vertex buffer
	uint32_t *indices; // unused — index buffer is bound separately
	uint32_t textureHandle;
	int width;
	int height;
	float3 compressionFactor;
	float min_x;
	float min_y;
	float min_z;
	uint32_t index_min;
	uint32_t bitsPerIndex;
};

[[vk::push_constant]]
cbuffer PC
{
	column_major float4x4 viewProj;
	Node*                 nodes;
	uint32_t*             cumulativeTriCounts;
};

struct VertOut
{
	float4               pos   : SV_Position;
	nointerpolation uint visID : TEXCOORD0;
};

uint32_t readBits(uint32_t* data, uint32_t bitOffset, uint32_t bitCount)
{
	uint32_t wordIndex = bitOffset >> 5;
	uint32_t bitStart  = bitOffset & 31;
	uint32_t word      = data[wordIndex];

	if (bitStart + bitCount > 32)
	{
		uint32_t bitsInFirst  = 32 - bitStart;
		uint32_t bitsInSecond = bitCount - bitsInFirst;
		uint32_t lo = word >> bitStart;
		uint32_t hi = data[wordIndex + 1] & ((1u << bitsInSecond) - 1u);
		return lo | (hi << bitsInFirst);
	}

	uint32_t mask = (bitCount == 32) ? 0xFFFFFFFFu : ((1u << bitCount) - 1u);
	return (word >> bitStart) & mask;
}

[shader("vertex")]
VertOut main_vertex(uint vertexID : SV_VertexID, uint drawID : SV_DrawIndex)
{
	Node node = nodes[drawID];

	uint32_t index = node.indices[vertexID];
	float3 pos = node.position[index];

	VertOut o;
	o.pos   = mul(viewProj, mul(node.world, float4(pos, 1.0)));
	o.visID = cumulativeTriCounts[drawID] + vertexID / 3u;
	return o;
}

[shader("vertex")]
VertOut main_vertex_compressed(uint vertexID : SV_VertexID, uint drawID : SV_DrawIndex)
{
	Node node = nodes[drawID];

    uint32_t index = readBits(node.indices, node.bitsPerIndex * vertexID, node.bitsPerIndex) + node.index_min;
	vector<uint16_t, 3> pos_u16 = reinterpret<vector<uint16_t, 3>*>(node.position)[index];

	float3 pos;
	pos.x = float(pos_u16.x) * node.compressionFactor.x + node.min_x;
	pos.y = float(pos_u16.y) * node.compressionFactor.y + node.min_y;
	pos.z = float(pos_u16.z) * node.compressionFactor.z + node.min_z;

	VertOut o;
	o.pos   = mul(viewProj, mul(node.world, float4(pos, 1.0)));
	o.visID = cumulativeTriCounts[drawID] + vertexID / 3u;
	return o;
}

[shader("fragment")]
uint main_fragment(VertOut i) : SV_Target0
{
	return i.visID;
}
