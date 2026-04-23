
import * as fs from "fs";
import * as path from "path";

let gltfPath = "F:/resources/meshes/odm_wietrznia.glb";
// let gltfPath = "F:/resources/meshes/zorah_main_public.gltf/zorah_main_public.gltf";
// let gltfPath = "F:/resources/meshes/sponza-png_by_Ludicon.glb";
// let gltfPath = "F:/resources/meshes/donaukanal_urania.glb";

const COMPONENT_TYPE = {
	5120: "BYTE",
	5121: "UNSIGNED_BYTE",
	5122: "SHORT",
	5123: "UNSIGNED_SHORT",
	5125: "UNSIGNED_INT",
	5126: "FLOAT",
};

// Returns { gltf, binFd, binChunkDataOffset }
// binChunkDataOffset is 0 for separate .bin files (raw binary), or the
// offset into the .glb file where the BIN chunk data starts.
function loadGltf(filePath) {
	let ext = path.extname(filePath).toLowerCase();

	if (ext === ".glb") {
		let fd = fs.openSync(filePath, "r");

		// 12-byte file header
		let header = Buffer.alloc(12);
		fs.readSync(fd, header, 0, 12, 0);
		let magic   = header.readUInt32LE(0);
		let version = header.readUInt32LE(4);
		let length  = header.readUInt32LE(8);
		if (magic !== 0x46546C67) throw new Error(`Not a GLB file (magic = 0x${magic.toString(16)})`);
		console.log(`GLB version: ${version}, total length: ${length} bytes`);

		// JSON chunk header at offset 12
		let chunkHeader = Buffer.alloc(8);
		fs.readSync(fd, chunkHeader, 0, 8, 12);
		let jsonChunkLength = chunkHeader.readUInt32LE(0);
		let jsonChunkType   = chunkHeader.readUInt32LE(4);
		if (jsonChunkType !== 0x4E4F534A) throw new Error(`First chunk is not JSON`);

		// BIN chunk data starts after: 12 (file hdr) + 8 (JSON chunk hdr) + jsonChunkLength + 8 (BIN chunk hdr)
		let binChunkDataOffset = 20 + jsonChunkLength + 8;
		console.log(`JSON chunk offset: 20, length: ${jsonChunkLength} bytes`);
		console.log(`BIN chunk data offset: ${binChunkDataOffset}`);

		let jsonBuffer = Buffer.alloc(jsonChunkLength);
		fs.readSync(fd, jsonBuffer, 0, jsonChunkLength, 20);
		let gltf = JSON.parse(jsonBuffer.toString("utf8"));

		return { gltf, binFd: fd, binChunkDataOffset };

	} else if (ext === ".gltf") {
		let gltf = JSON.parse(fs.readFileSync(filePath, "utf8"));

		// Resolve the .bin file from the buffer URI (relative to the .gltf file)
		let bufferUri = gltf.buffers?.[0]?.uri;
		if (!bufferUri) throw new Error("No buffer URI found in .gltf");
		let binPath = path.resolve(path.dirname(filePath), bufferUri);
		console.log(`Loading binary buffer: ${binPath}`);

		let binFd = fs.openSync(binPath, "r");
		return { gltf, binFd, binChunkDataOffset: 0 };

	} else {
		throw new Error(`Unsupported file extension: ${ext}`);
	}
}

let { gltf, binFd, binChunkDataOffset } = loadGltf(gltfPath);

let indexBuffers = [];

for (let mi = 0; mi < (gltf.meshes?.length ?? 0); mi++) {
	let mesh = gltf.meshes[mi];
	for (let pi = 0; pi < (mesh.primitives?.length ?? 0); pi++) {
		let prim = mesh.primitives[pi];
		if (prim.indices == null) continue;

		let acc = gltf.accessors[prim.indices];
		let bv  = gltf.bufferViews[acc.bufferView];

		indexBuffers.push({
			mesh:          mi,
			meshName:      mesh.name ?? "(unnamed)",
			primitive:     pi,
			accessorIdx:   prim.indices,
			count:         acc.count,
			componentType: COMPONENT_TYPE[acc.componentType] ?? acc.componentType,
			bufferView:    acc.bufferView,
			byteOffset:    (bv.byteOffset ?? 0) + (acc.byteOffset ?? 0),
			byteLength:    bv.byteLength,
		});
	}
}

console.log(`Found ${indexBuffers.length} index buffer(s):\n`);
for (let ib of indexBuffers) {
	let numIndices = ib.count;

	let bytesPerIndex = ib.componentType === "UNSIGNED_SHORT" ? 2 : 4;
	let fileOffset = binChunkDataOffset + ib.byteOffset;
	let data = Buffer.alloc(bytesPerIndex * numIndices);
	fs.readSync(binFd, data, 0, data.byteLength, fileOffset);

	let readIndex = bytesPerIndex === 2
		? (i) => data.readUInt16LE(2 * i)
		: (i) => data.readUInt32LE(4 * i);

	let uniqueIndicesSet = new Set();
	let n = Math.min(51200, numIndices);
	for (let i = 0; i < n; i++) uniqueIndicesSet.add(readIndex(i));
	let uniqueIndices = Array.from(uniqueIndicesSet);

	let uniqueRatio = uniqueIndices.length / n;
	let reuseRatio = 1 - uniqueRatio;
	let numTriangles = ib.count / 3;
	console.log(`#triangles: ${numTriangles.toLocaleString().padStart(10)} indices: ${n}, unique: ${uniqueIndices.length}, reuse: ${(100 * reuseRatio).toFixed(1)} % `);
}

fs.closeSync(binFd);
