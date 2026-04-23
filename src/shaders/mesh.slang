// mesh_vertexbuffer.slang — Textured mesh rendering (vertex buffers + index buffer, instanced)
// slangc mesh.slang -target spirv -fvk-use-entrypoint-name -capability spvBindlessTextureNV -source-embed-style u32 -source-embed-name mesh_shaders_indexedDraw_spv -o mesh.h

// Must match VKNode exactly.
struct Node {
	column_major float4x4 world;
	column_major float4x4 worldViewProj;
	float3*               position;         // unused — position comes from vertex buffer
	float2*               uv;               // unused — uv comes from vertex buffer
	uint32_t*             indices;          // unused — index buffer is bound separately
	uint32_t              textureHandle;
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
	Node*                 nodes;      // BDA — one Node per instance
};

struct VertOut
{
	float4               pos       : SV_Position;
	float2               uv        : TEXCOORD0;
	nointerpolation uint texHandle : TEXCOORD1;
};

[shader("vertex")]
VertOut main_vertex(
	[vk::location(0)] float3 position : POSITION,
	[vk::location(1)] float2 uv_in    : TEXCOORD0,
	uint instanceID : SV_InstanceID,
	uint firstInstance : SV_StartInstanceLocation 
){
	Node node   = nodes[firstInstance + instanceID];

	VertOut o;
    // o.pos       = mul(viewProj, mul(node.world, float4(position, 1.0)));
    o.pos       = mul(node.worldViewProj, float4(position, 1.0));
	o.uv        = uv_in;
	o.texHandle = node.textureHandle;

	return o;
}

[shader("fragment")]
float4 main_fragment(VertOut i) : SV_Target0
{
	Sampler2D tex = Sampler2D.Handle(int(i.texHandle));
	return tex.Sample(i.uv);
}
