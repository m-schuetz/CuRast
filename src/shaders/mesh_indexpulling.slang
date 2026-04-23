// mesh.slang — Textured mesh rendering
// slangc mesh_indexpulling.slang -target spirv -fvk-use-entrypoint-name -capability spvBindlessTextureNV -source-embed-style u32 -source-embed-name mesh_shaders_indexpulling_spv -o mesh_indexpulling.h

struct Node
{
    column_major float4x4 world;
    column_major float4x4 worldViewProj;
	float3*               position;
	float2*               uv;           // BDA — float2 array (numVertices entries)
	uint32_t*             indices;      // BDA — uint32 index array (numTriangles*3 entries)
	uint32_t              textureHandle; // VK_NVX_image_view_handle — combined image sampler handle
	int                   width;
	int                   height;
	float3                compressionFactor;
	float                 min_x;
	float                 min_y;
	float                 min_z;
	uint32_t              index_min;
	uint32_t              bitsPerIndex;
};

[[vk::push_constant]]
cbuffer PC
{
	column_major float4x4 viewProj;
	Node*                 nodes; // BDA — Node[]
};

struct VertOut
{
	float4               pos        : SV_Position;
	float2               uv         : TEXCOORD0;
	nointerpolation uint texHandle  : TEXCOORD1;
	float                depth;
};

uint32_t readBits(uint32_t* data, uint32_t  bitOffset, uint32_t  bitCount) {
	uint32_t wordIndex = bitOffset >> 5;          // divide by 32
	uint32_t bitStart  = bitOffset & 31;          // modulo 32

	uint32_t word = data[wordIndex];
	uint32_t mask = (bitCount == 32) ? 0xFFFFFFFFu : ((1u << bitCount) - 1u);

	// If the bits span two words
	if (bitStart + bitCount > 32)
	{
		uint32_t bitsInFirst  = 32 - bitStart;
		uint32_t bitsInSecond = bitCount - bitsInFirst;

		uint32_t lo = word >> bitStart;
		uint32_t hi = data[wordIndex + 1] & ((1u << bitsInSecond) - 1u);
		return lo | (hi << bitsInFirst);
	}

	return (word >> bitStart) & mask;
}

[shader("vertex")]
VertOut main_vertex_instanced(
	uint vertexID   : SV_VertexID,
	uint instanceID : SV_InstanceID,
	uint firstInstance : SV_StartInstanceLocation 
){
	Node node = nodes[firstInstance + instanceID];

	uint32_t index = node.indices[vertexID];
	float3 pos = node.position[index];
	
	VertOut o;
	o.pos       = mul(node.worldViewProj, float4(pos, 1.0));
	if(node.uv){
		o.uv        = node.uv[index];
		o.texHandle = node.textureHandle;
	}else{
		o.texHandle = 0xffffffff;
	}

	o.depth = o.pos.w;

	return o;
}

[shader("vertex")]
VertOut main_vertex_compressed_instanced(
	uint vertexID   : SV_VertexID,
	uint instanceID : SV_InstanceID,
	uint firstInstance : SV_StartInstanceLocation 
){
	Node node = nodes[firstInstance + instanceID];

	uint32_t index = readBits(node.indices, node.bitsPerIndex * vertexID, node.bitsPerIndex) + node.index_min;
	vector<uint16_t, 3> pos_u16 = reinterpret<vector<uint16_t, 3>*>(node.position)[index];

	float3 pos;
	pos.x = float(pos_u16.x) * node.compressionFactor.x + node.min_x;
	pos.y = float(pos_u16.y) * node.compressionFactor.y + node.min_y;
	pos.z = float(pos_u16.z) * node.compressionFactor.z + node.min_z;

	VertOut o;
	o.pos = mul(node.worldViewProj, float4(pos, 1.0));
	if(node.uv){
		o.uv        = node.uv[index];
		o.texHandle = node.textureHandle;
	}else{
		o.texHandle = 0xffffffff;
	}

	o.depth = o.pos.w;

	return o;
}

[shader("fragment")]
float4 main_fragment(VertOut fragment) : SV_Target0
{

	if(fragment.texHandle != 0xffffffff){
		// float2 uv = clamp(fmod(fragment.uv, 1.0), 0.0, 1.0);
		// uv.x = uv.x - floor(uv.x);
		// uv.y = uv.y - floor(uv.y);
		Sampler2D tex = Sampler2D.Handle(int(fragment.texHandle));
		return tex.Sample(fragment.uv);

		//return float4(uv.x, uv.y, 0.0, 1.0);
	} else {
		float w = fragment.depth / 100.0f;
		return float4(w, w, w, 1.0f);
	}

}