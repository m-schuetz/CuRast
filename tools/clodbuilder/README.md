# clodbuilder

Converts a GLB mesh into a clustered LOD representation (a Nanite-style cluster DAG). The clusters, groups, simplification errors and the group hierarchy are computed by [meshoptimizer](https://github.com/zeux/meshoptimizer) v1.3, using `clodBuild()` and `clodBuildHierarchy()` from its [demo/clusterlod.h](https://github.com/zeux/meshoptimizer/blob/v1.3/demo/clusterlod.h). This tool only loads the GLB and writes the results into flat binary files that can be memory-mapped or copied to the GPU as they are.

Supported input: one mesh with one triangle primitive with `POSITION`, `TEXCOORD_0` and indices, no node transformations, and optionally a base color texture embedded in the GLB.

## Build and run

meshoptimizer is downloaded by CMake at configure time.

```
cmake -S tools/clodbuilder -B tools/clodbuilder/build -DCMAKE_BUILD_TYPE=Release
cmake --build tools/clodbuilder/build -j
./tools/clodbuilder/build/clodbuilder <input.glb> <outputDir>
```

CuRast reads the texture from a BC7-compressed `texture.dds` with mip levels. Convert the `texture.jpg` that clodbuilder writes with [AMD Compressonator](https://github.com/GPUOpen-Tools/compressonator/releases) (V4.5.52, CLI for Linux). Let the levels end at 4x4: Compressonator V4.5.52 encodes the 2x2 level incorrectly (half of its texels are black). For an 8192x8192 texture, that's 12 levels:

```
compressonatorcli -fd BC7 -miplevels 12 -NumThreads 32 <outputDir>/texture.jpg <outputDir>/texture.dds
```

## Settings

`clodDefaultConfig(128)`, meshoptimizer's default for rasterization: at most 128 triangles and 128 vertices per cluster, groups of about 16 clusters, each level aims to halve the triangle count. Two settings differ from the defaults:

- `partition_sort = true`: groups within a level are sorted spatially, so neighboring groups are stored next to each other.
- `optimize_bounds = true`: `cullSphere` tightly encloses each cluster's triangles.

UVs are not weighted during simplification, but UV seams are protected (`attribute_protect_mask = 0b11`). meshoptimizer's own `demo/nanite.cpp` uses the same setup. If simplification gets stuck, clodBuild falls back to sloppy simplification and doubles the error of the result.

## Output files

All values are little-endian. Clusters are stored in the order clodBuild produces them: by level, finest first, with the clusters of each group stored contiguously. Each cluster's vertices and triangles are contiguous ranges in the vertex and triangle files. Vertices on cluster borders are duplicated, so each cluster can be read on its own.

| File | Content |
|------|---------|
| `metadata.json` | Counts, bounding box, per-level statistics, the clodBuild settings, and the file list with record sizes |
| `clusters.bin` | `Cluster[numClusters]`, 112 bytes each |
| `groups.bin` | `Group[numGroups]`, 32 bytes each |
| `nodes.bin` | `Node[numNodes]`, 32 bytes each: a BVH over the groups for hierarchical LOD selection and culling |
| `positions.bin` | `float x, y, z` per vertex, in the units of the source mesh |
| `uvs.bin` | `float u, v` per vertex, glTF convention: (0, 0) is the top-left corner of the texture |
| `triangles.bin` | `uint8 i0, i1, i2` per triangle, indices relative to the cluster's `vertexOffset` |
| `texture.jpg` | The source's base color texture, copied as is |
| `texture.dds` | Not written by clodbuilder: The texture, BC7-compressed with mip levels (see above) |

```cpp
struct Cluster{                 // 112 bytes
	float    aabbMin[3];
	uint32_t vertexOffset;      // first vertex in positions.bin and uvs.bin
	float    aabbMax[3];
	uint32_t triangleOffset;    // first triangle in triangles.bin (byte offset = 3 * triangleOffset)
	float    cullSphere[4];     // tight bounding sphere of this cluster's triangles. xyz: center, w: radius
	float    lodSphere[4];      // bounds of the group this cluster was simplified from
	float    parentSphere[4];   // bounds of the group this cluster belongs to
	float    lodError;          // 0 for clusters of the original mesh
	float    parentError;       // FLT_MAX if this cluster's group was not simplified any further
	uint32_t vertexCount;       // <= 128
	uint32_t triangleCount;     // <= 128
	uint32_t level;             // DAG depth of this cluster's group, 0 = original mesh
	uint32_t group;             // the group this cluster belongs to
	int32_t  refinedGroup;      // the group this cluster was simplified from, -1 for clusters of the original mesh
	uint32_t padding;
};

struct Group{                   // 32 bytes
	float    sphere[4];         // bounds of the group's simplified result. xyz: center, w: radius
	float    error;             // simplification error of that result; FLT_MAX if the group is terminal
	uint32_t level;
	uint32_t clusterOffset;     // first cluster of this group in clusters.bin
	uint32_t clusterCount;
};

struct Node{                    // 32 bytes, identical to clodNode
	float    sphere[4];         // encloses all groups in this subtree
	float    error;             // maximum error of all groups in this subtree
	int32_t  group;             // leaf: index into groups.bin, internal node: -1
	uint32_t childOffset;       // internal node: children are nodes[childOffset, childOffset + childCount)
	uint32_t childCount;
};
```

Errors are in the units of the source mesh. `lodSphere`/`lodError` equal the `Group` record of `refinedGroup`, and `parentSphere`/`parentError` equal the `Group` record of `group`. They are copied into each cluster so that a cluster can be tested without looking up its groups.

## LOD selection

Project an error onto the screen with the bounding sphere it belongs to. `proj` is `projection[1][1]`, i.e. `1 / tan(fovy / 2)`. The result is a fraction of the screen height, so multiply by the framebuffer height to get pixels.

```cpp
float projectedError(vec3 center, float radius, float error, vec3 cameraPos, float proj, float znear){
	float d = max(distance(center, cameraPos) - radius, znear);
	return error / d * proj * 0.5;
}
```

For a threshold `t` (e.g. 1 pixel: `t = 1.0 / framebufferHeight`), a cluster is rendered if

```cpp
projectedError(cluster.lodSphere,    cluster.lodError)    <= t &&
projectedError(cluster.parentSphere, cluster.parentError) >  t
```

Every cluster of a group, and every cluster made from the same refined group, uses the same sphere and error. Neighboring clusters therefore always make the same decision, and the selected clusters form a crack-free cut through the DAG. The rule can be evaluated for all clusters independently, e.g. one thread per cluster.

`nodes.bin` lets you avoid testing every cluster. It is a forest with one tree per level, and the root of level `i` is `nodes[i]`. Starting at each root, descend into a node only if `projectedError(node.sphere, node.error) > t`. On reaching a leaf, render each cluster of `groups[leaf.group]` whose `refinedGroup` is -1 or whose `projectedError(lodSphere, lodError) <= t`. This gives exactly the same clusters as testing each cluster. Frustum culling can use `cullSphere` or the AABB for clusters, and `Node.sphere` for subtrees.
