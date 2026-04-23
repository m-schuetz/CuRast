
import * as fs from "fs";
import { buffer } from "stream/consumers";

let gltfPath = "F:/resources/meshes/zorah_main_public.gltf/zorah_main_public.gltf";
let gltfDataPath = "F:/resources/meshes/zorah_main_public.gltf/zorah_main_public.gltf.bin";

let data = fs.readFileSync(gltfPath, "ascii");
// let data = fs.readFileSync("D:/dev/workspaces/IVA/temp/donaukanal.json", "ascii");
let gltf = JSON.parse(data);

// console.log(gltf);


let mesh_accessorIndices = [];
let mesh_bufferViews = [];

for(let mesh of gltf.meshes){
	let accessorIndices = [];

	for(let primitive of mesh.primitives){
		accessorIndices.push(primitive.indices);

		for(let attributeName of Object.keys(primitive.attributes)){
			accessorIndices.push(primitive.attributes[attributeName]);
		}
	}

	let bufferViewIndices = [];
	for(let accessorIndex of accessorIndices){
		let accessor = gltf.accessors[accessorIndex];
		bufferViewIndices.push(accessor.bufferView);
	}

	mesh_accessorIndices.push(accessorIndices);
	mesh_bufferViews.push(bufferViewIndices);
}

// Check if there are multiple uses of a buffer view over different meshes
for(let bufferViewIndex = 0; bufferViewIndex < Math.min(gltf.bufferViews.length, 20); bufferViewIndex++){

	let occurances = [];

	for(let meshIndex = 0; meshIndex < gltf.meshes.length; meshIndex++){
		let bufferViewIndices = mesh_bufferViews[meshIndex];
		for(let meshBufferViewIndex of bufferViewIndices){
			if(meshBufferViewIndex === bufferViewIndex){
				occurances.push(meshIndex);
			}
		}
	}

	let set = new Set();

	for(let occurance of occurances){
		set.add(occurance);
	}

	if(set.size > 0){
		console.log(`BufferView ${bufferViewIndex} occurs in meshes ${Array.from(set)}`);
	}
}


{ // Check number of triangles per mesh
	let format = new Intl.NumberFormat("en-US");
	let numTrianglesList = [];
	for(let mesh of gltf.meshes){
		
		let numTrianglesInMesh = 0;
		for(let primitive of mesh.primitives){
			
			let accessor = gltf.accessors[primitive.indices];
			let numIndices = accessor.count;
			let numTriangles = numIndices / 3;
			
			numTrianglesInMesh += numTriangles;
		}

		numTrianglesList.push(numTrianglesInMesh);
		// console.log(format.format(numTrianglesInMesh).padStart(10));
	}

	numTrianglesList.sort((a, b) => a - b);
	let maxNumTriangles = Math.max(...numTrianglesList);

	let binSize = 5_000_000;
	let binCount = Math.ceil(maxNumTriangles / binSize);
	let bins = new Array(binCount).fill(0);

	for(let numTriangles of numTrianglesList){
		let binIndex = Math.floor(numTriangles / binSize);
		bins[binIndex]++;
	}

	for(let i = 0; i < bins.length; i++){
		let M = (i * binSize) / 1_000_000;
		let Mn = ((i + 1) * binSize) / 1_000_000;
		let key = `${M.toString().padStart(3)} - ${Mn.toString().padStart(3)} million triangles`;
		console.log(`${key}: ${bins[i].toString().padStart(5)}`);
	}
}


// Print stats for large meshes
let largeMeshes = [];
for(let mesh of gltf.meshes){

	for(let primitive of mesh.primitives){
		let accessor = gltf.accessors[primitive.indices];
		let numIndices = accessor.count;
		let numTriangles = numIndices / 3;

		if(numTriangles > 10_000_000){

			largeMeshes.push(mesh);

			// for(let attributeName of Object.keys(primitive.attributes)){
			// 	accessorIndices.push(primitive.attributes[attributeName]);
			// }

			let accessorIndex_pos = primitive.attributes.POSITION;
			let aPos = gltf.accessors[accessorIndex_pos];
			let numVertices = aPos.count;

			let size = [
				aPos.max[0] - aPos.min[0],
				aPos.max[1] - aPos.min[1],
				aPos.max[2] - aPos.min[2],
			];
			let precision = 0.0001;

			let requiredBitsPosition = [
				Math.ceil(Math.log2(size[0] / precision)),
				Math.ceil(Math.log2(size[1] / precision)),
				Math.ceil(Math.log2(size[2] / precision)),
			];
			let sumRequiredBitsPosition = requiredBitsPosition[0] + requiredBitsPosition[1] + requiredBitsPosition[2];

			let requiredBitsPerIndex = Math.ceil(Math.log2(numTriangles));

			let totalRequieredBits = numVertices * sumRequiredBitsPosition 
				+ 3 * requiredBitsPerIndex * numTriangles
				+ 8 * numVertices;
			let bytesPerTriangle_compressed = (totalRequieredBits / 8) / numTriangles;
			let bytesPerTriangle = (12 * numVertices + 8 * numVertices + 3 * 4 * numTriangles) / numTriangles;

			console.log(`# large mesh, #triangles: ${(numTriangles / 1_000_000).toFixed(1)} M`);
			console.log("    size: ", size.map(v => v.toFixed(3)).join(", "));
			console.log("    required bits for position: ", requiredBitsPosition.join(", "), "sum: ", sumRequiredBitsPosition);
			console.log("    required bits per index: ", requiredBitsPerIndex);
			console.log("    bytes per triangle:              ", bytesPerTriangle.toFixed(1));
			console.log("    bytes per triangle (compressed): ", bytesPerTriangle_compressed.toFixed(1));
		}
	}
}

// check index ranges

function checkIndexRanges(){
	let totalSavedBytes = 0;

	let total_bytes_indices_u32 = 0;
	let total_bytes_indices_compressed_u16 = 0;
	let total_bytes_indices_compressed_u32 = 0;
	let total_bytes_indices_compressed_maxrange = 0;

	for(let meshIndex = 0; meshIndex < gltf.meshes.length; meshIndex++){
		let mesh = gltf.meshes[meshIndex];
		
		for(let primitiveIndex = 0; primitiveIndex < mesh.primitives.length; primitiveIndex++){

			let primitive = mesh.primitives[primitiveIndex];
			let accessor = gltf.accessors[primitive.indices];
			let view = gltf.bufferViews[accessor.bufferView];
			let numTriangles = accessor.count / 3;

			// if(numTriangles < 1_000_000) continue;
			
			const fd = fs.openSync(gltfDataPath, "r");
			const buffer = Buffer.alloc(view.byteLength);
			fs.readSync(fd, buffer, 0, view.byteLength, view.byteOffset);
			fs.closeSync(fd);

			let index_min = Infinity;
			let index_max = 0;
			let largestIndexSpan = 0;
			let bits = {};
			let maxBitsPerIndex = 0;
			for(let triangleIndex = 0; triangleIndex < numTriangles; triangleIndex++){

				let i0 = buffer.readUint32LE(12 * triangleIndex + 0);
				let i1 = buffer.readUint32LE(12 * triangleIndex + 4);
				let i2 = buffer.readUint32LE(12 * triangleIndex + 8);

				index_min = Math.min(index_min, i0, i1, i2);
				index_max = Math.max(index_max, i0, i1, i2);
				
				if((triangleIndex % 128) === 127 || triangleIndex === (numTriangles - 1)){
					let index_span = index_max - index_min;
					let bitsPerIndex = Math.ceil(Math.log2(index_span));
					maxBitsPerIndex = Math.max(maxBitsPerIndex, bitsPerIndex);

					// bits.push(bitsPerIndex);
					if(!bits[bitsPerIndex]){
						bits[bitsPerIndex] = 0;
					}
					bits[bitsPerIndex]++;

					if(index_span > largestIndexSpan){
						largestIndexSpan = index_span;
					}

					index_min = Infinity;
					index_max = 0;
				}
			}

			let uncompressedBitCount = 3 * numTriangles * 32;
			let compressedBitCount = 3 * numTriangles * maxBitsPerIndex;
			let compressionRate = compressedBitCount / uncompressedBitCount;
			let savedBytes = (uncompressedBitCount - compressedBitCount) / 8;
			totalSavedBytes += savedBytes;

			// console.log(`mesh: ${meshIndex}, primitive: ${primitiveIndex}, #triangles: ${numTriangles.toLocaleString()}, maxBitsPerIndex: ${maxBitsPerIndex}, savedBytes: ${savedBytes}`);

			let keys = Object.keys(bits);
			let msg = keys.map(key => `${key}: ${bits[key]}`).join(", ");
			// console.log(msg);
			// console.log(bits);

			let numle16 = 0;
			let numg16 = 0;
			for(let i = 0; i < 32; i++){
				if(bits[i]){
					if(i <= 16) numle16 += bits[i];
					if(i  > 16) numg16 += bits[i];
				}
			}

			let bytes_uncompressed = numTriangles * 3 * 4;
			let bytes16 = numle16 * 128 * 3 * 2;
			let bytes32 = numg16 * 128 * 3 * 4;

			total_bytes_indices_u32 += bytes_uncompressed;
			total_bytes_indices_compressed_u16 += bytes16;
			total_bytes_indices_compressed_u32 += bytes32;
			total_bytes_indices_compressed_maxrange += Math.ceil(compressedBitCount / 8);

			// console.log({numle16, numg16, bytes16, bytes32});
		}
	}
	// console.log(`totalSavedBytes: ${totalSavedBytes.toLocaleString()}`);
	console.log(`total_bytes_indices_u32:                  ${total_bytes_indices_u32.toLocaleString().padStart(15)}`);
	console.log(`total_bytes_indices_compressed_u16:       ${total_bytes_indices_compressed_u16.toLocaleString().padStart(15)}`);
	console.log(`total_bytes_indices_compressed_u32:       ${total_bytes_indices_compressed_u32.toLocaleString().padStart(15)}`);
	console.log(`total_bytes_indices_compressed_maxrange:  ${total_bytes_indices_compressed_maxrange.toLocaleString().padStart(15)}`);
}

// for(let meshIndex = 0; meshIndex < gltf.meshes.length; meshIndex++){
// 	let mesh = gltf.meshes[meshIndex];

// 	if(meshIndex === 2023) console.log(mesh.primitives[0]);
	
// 	for(let primitiveIndex = 0; primitiveIndex < mesh.primitives.length; primitiveIndex++){

// 		let primitive = mesh.primitives[primitiveIndex];

// 		let numTriangles = gltf.accessors[primitive.indices].count / 3;
		
// 		let accessorIndex = primitive.attributes["POSITION"];
// 		let accessor = gltf.accessors[accessorIndex];
// 		let numVertices = accessor.count;

// 		if(numTriangles > 20_000_000)
// 		console.log({meshIndex, primitiveIndex, numTriangles, numVertices});
// 	}
// }

console.log(gltf.accessors[5909]);
console.log(gltf.bufferViews[4125]);

// How many bytes for positions?

function checkBytesAndRangeIntersections(){
	let accessedRanges = [];

	function intersectsRange(newRange){

		for(let range of accessedRanges){
			if(newRange.byteOffset >= range.byteOffset + range.byteSize) continue;
			if(newRange.byteOffset + newRange.byteSize < range.byteOffset) continue;

			console.log("Ranges intersect!");
			console.log(range);
			console.log(newRange);
			return true;
		}

		return false;
	}

	let numPositionBytes = 0;
	let numUVBytes = 0;
	let numIndicesBytes = 0;

	for(let accessorIndex = 0; accessorIndex < gltf.accessors.length; accessorIndex++){
		let accessor = gltf.accessors[accessorIndex];
		let bufferView = gltf.bufferViews[accessor.bufferView];
		let byteOffset = bufferView.byteOffset + accessor.byteOffset;

		if(accessor.type === "VEC3"){
			let byteSize = accessor.count * 12;
			let range = {accessorIndex, byteOffset, byteSize};

			numPositionBytes += byteSize;

			if(intersectsRange(range)){
				return;
			}

			accessedRanges.push(range);
		}else if(accessor.type === "VEC2"){
			let byteSize = accessor.count * 8;
			let range = {accessorIndex, byteOffset, byteSize};

			numUVBytes += byteSize;

			if(intersectsRange(range)){
				return;
			}

			accessedRanges.push(range);
		}else if(accessor.type === "SCALAR" && accessor.componentType === 5125){
			let byteSize = accessor.count * 4;
			let range = {accessorIndex, byteOffset, byteSize};

			numIndicesBytes += byteSize;

			if(intersectsRange(range)){
				return;
			}

			accessedRanges.push(range);
		}
	}

	console.log("bytes_position: ", (numPositionBytes).toLocaleString().padStart(15));
	console.log("bytes_uv:       ", (numUVBytes).toLocaleString().padStart(15));
	console.log("bytes_indices   ", (numIndicesBytes).toLocaleString().padStart(15));
	console.log("=".repeat(30));
	console.log("total           ", (numPositionBytes + numUVBytes + numIndicesBytes).toLocaleString().padStart(15));
}

checkIndexRanges();
checkBytesAndRangeIntersections();

9_513_192_446 + 17_129_858_376 / 2