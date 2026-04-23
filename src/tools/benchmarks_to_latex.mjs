import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

const BENCHMARKS_DIR = "D:/dev/workspaces/IVA/benchmarks";
// const BENCHMARKS_DIR = "I:/IVA/benchmarks";



const files = fs.readdirSync(BENCHMARKS_DIR).filter(f => f.endsWith('.json'));

const benchmarks = files.map(f =>
	JSON.parse(fs.readFileSync(path.join(BENCHMARKS_DIR, f), 'utf8'))
);

const rowMap = new Map();

function benchKey(benchmark){

	let desc = {
		scene:               benchmark.scene,
		compressed:          benchmark.hasCompressedGeometry ?? false,
		jpeg:                benchmark.usesJpegTextures ?? false,
		path:                benchmark.path,
		imageDivisionFactor: benchmark.imageDivisionFactor ?? 1,
	};

	let key = JSON.stringify(desc);

	return key;
}

// Sanitize/Normalize input
for (const benchmark of benchmarks) {
	benchmark.usesJpegTextures = benchmark.usesJpegTextures ?? false;
}

for (const benchmark of benchmarks) {

	let key = benchKey(benchmark);

	if(!rowMap.has(key)){
		rowMap.set(key, []);
	}

	rowMap.get(key).push(benchmark);
}

let rows = Array.from(rowMap.keys());

const ORDERING = {
	"Sponza" :  0,
	"Lantern": 10,
	"Lantern Instances": 11,
	"Japan Statue" : 15,
	"Komainu Kobe" : 15,
	"Venice" : 20,
	"Zorah"  : 30,
};

rows.sort( (a, b) => {
	let a_scene = JSON.parse(a).scene;
	let b_scene = JSON.parse(b).scene;

	if(!a_scene) console.log(ORDERING[a_scene]);

	return ORDERING[a_scene] - ORDERING[b_scene];
});


function findItem(items, gpu, activeView, method){

	for(let item of items){

		if(!item.device.includes(gpu)) continue;
		if(item.method !== method) continue;
		if(item.activeView !== activeView) continue;

		return item;
	}

	return null;
}


console.log("==================================================================================================");
console.log(`
\\begin{table*}[t]
\\centering
\\caption{Benchmark timings in milliseconds. \\textbf{c}: Compressed coordinates and index buffers. \\textbf{m}: Meshes optimized with meshoptimizer/gltfpack~\\cite{meshoptimizer}. \\textbf{j}: JPEG-compressed textures. \\textbf{h}: Resized texture to half resolution. \\textbf{--}: Out of memory. \\textbf{$\\times$}: Unsupported test configuration. E.g., Vulkan indexed draws do not support compressed index buffers, and JPEG-compressed textures are only supported in our CUDA rasterizer.}
\\label{tab:benchmark_results}
%\\begin{tabularx}{\\textwidth}{l l r l l l l *{9}{Y}}
% >{\\hspace{10pt}}r<{\\hspace{10pt}}
\\rowcolors{1}{white}{gray!10}
\\begin{tabularx}{\\textwidth}{
	l<{\\hspace{5pt}} 
	l<{\\hspace{10pt}} 
	r<{\\hspace{10pt}} 
	l l l l 
	*{9}{Y}}
	\\toprule
	& & & & & &
	& \\multicolumn{3}{c}{RTX 4070}
	& \\multicolumn{3}{c}{RTX 4090}
	& \\multicolumn{3}{c}{RTX 5090} \\\\
	\\cmidrule(lr){8-10} \\cmidrule(lr){11-13} \\cmidrule(lr){14-16}
	& Scene & vis. tris & {} & {} & {} &
	& {CuRast} & {VK-ID} & {VK-PIP}
	& {CuRast} & {VK-ID} & {VK-PIP}
	& {CuRast} & {VK-ID} & {VK-PIP} \\\\
	\\midrule
`);


for(let view of ["closeup", "overview"]){

	console.log(`\t%`);
	console.log(`\t% ${view}`);
	console.log(`\t%`);

	for(let r = 0; r < rows.length; r++){

		let row = rows[r];
		let items = rowMap.get(row);

		let b_4070_cuda   = findItem(items, "RTX 4070", view, "CUDA_VISBUFFER_INSTANCED");
		let b_4070_vk_ind = findItem(items, "RTX 4070", view, "VULKAN_INDEXED_DRAW");
		let b_4070_vk_IP  = findItem(items, "RTX 4070", view, "VULKAN_INDEXPULLING_INSTANCED");

		let b_4090_cuda   = findItem(items, "RTX 4090", view, "CUDA_VISBUFFER_INSTANCED");
		let b_4090_vk_ind = findItem(items, "RTX 4090", view, "VULKAN_INDEXED_DRAW");
		let b_4090_vk_IP  = findItem(items, "RTX 4090", view, "VULKAN_INDEXPULLING_INSTANCED");

		let b_5090_cuda   = findItem(items, "RTX 5090", view, "CUDA_VISBUFFER_INSTANCED");
		let b_5090_vk_ind = findItem(items, "RTX 5090", view, "VULKAN_INDEXED_DRAW");
		let b_5090_vk_IP  = findItem(items, "RTX 5090", view, "VULKAN_INDEXPULLING_INSTANCED");

		let flags_c = items[0].hasCompressedGeometry ? "c" : " ";
		let flags_m = items[0].path.includes("_optimized") ? "m":  " ";
		let flags_j = items[0].usesJpegTextures ? "j" : " ";
		let flags_h = (items[0].imageDivisionFactor ?? 1) === 1 ? " " : "h";

		// We simplified the geometry of the lantern data set with meshoptimizer.
		if(items[0].scene === "Lantern" || items[0].scene === "Lantern Instances"){
			flags_m = "m";
		}

		let duration_4070_cuda   = b_4070_cuda   ? (b_4070_cuda.durations["<triangles visbuffer pipeline>"] + b_4070_cuda.durations["kernel_resolve_visbuffer_to_colorbuffer2D"]) : null;
		let duration_4070_vk_id  = b_4070_vk_ind ? (b_4070_vk_ind.durations["vulkan drawIndexed"]) : null;
		let duration_4070_vk_pip = b_4070_vk_IP  ? (b_4070_vk_IP.durations["vulkan draw"]) : null;
		let duration_4090_cuda   = b_4090_cuda   ? (b_4090_cuda.durations["<triangles visbuffer pipeline>"] + b_4090_cuda.durations["kernel_resolve_visbuffer_to_colorbuffer2D"]) : null;
		let duration_4090_vk_id  = b_4090_vk_ind ? (b_4090_vk_ind.durations["vulkan drawIndexed"]) : null;
		let duration_4090_vk_pip = b_4090_vk_IP  ? (b_4090_vk_IP.durations["vulkan draw"]) : null;
		let duration_5090_cuda   = b_5090_cuda   ? (b_5090_cuda.durations["<triangles visbuffer pipeline>"] + b_5090_cuda.durations["kernel_resolve_visbuffer_to_colorbuffer2D"]) : null;
		let duration_5090_vk_id  = b_5090_vk_ind ? (b_5090_vk_ind.durations["vulkan drawIndexed"]) : null;
		let duration_5090_vk_pip = b_5090_vk_IP  ? (b_5090_vk_IP.durations["vulkan draw"]) : null;

		let toString = (value, a, b, c, benchmark, gpu) => {

			let fastest = Math.min(...[a, b, c].filter(v => v !== null));
			let slowest = Math.max(...[a, b, c].filter(v => v !== null));

			let result = "";

			if(value){
				if(value === fastest) result = `\\fa{${value.toFixed(3)}}`;
				else if(value === slowest) result = `\\sl{${value.toFixed(3)}}`;
				else result = `${value.toFixed(3)}`;
			}else{
				if(gpu === "RTX 4070") result = `--`;
				else if(gpu === "RTX 4090") result = `$\\times$`;
				else if(gpu === "RTX 5090") result = `$\\times$`;
				else result = "?????";
			}

			result = result.padStart(13);

			return result;
		};

		let str_4070_cuda   = toString(duration_4070_cuda, duration_4070_cuda, duration_4070_vk_id, duration_4070_vk_pip, b_4070_cuda, "RTX 4070");
		let str_4070_vk_id  = toString(duration_4070_vk_id, duration_4070_cuda, duration_4070_vk_id, duration_4070_vk_pip, b_4070_vk_ind, "RTX 4070");
		let str_4070_vk_pip = toString(duration_4070_vk_pip, duration_4070_cuda, duration_4070_vk_id, duration_4070_vk_pip, b_4070_vk_IP, "RTX 4070");

		let str_4090_cuda   = toString(duration_4090_cuda, duration_4090_cuda, duration_4090_vk_id, duration_4090_vk_pip, b_4090_cuda, "RTX 4090");
		let str_4090_vk_id  = toString(duration_4090_vk_id, duration_4090_cuda, duration_4090_vk_id, duration_4090_vk_pip, b_4090_vk_ind, "RTX 4090");
		let str_4090_vk_pip = toString(duration_4090_vk_pip, duration_4090_cuda, duration_4090_vk_id, duration_4090_vk_pip, b_4090_vk_IP, "RTX 4090");

		let str_5090_cuda   = toString(duration_5090_cuda, duration_5090_cuda, duration_5090_vk_id, duration_5090_vk_pip, b_5090_cuda, "RTX 5090");
		let str_5090_vk_id  = toString(duration_5090_vk_id, duration_5090_cuda, duration_5090_vk_id, duration_5090_vk_pip, b_5090_vk_ind, "RTX 5090");
		let str_5090_vk_pip = toString(duration_5090_vk_pip, duration_5090_cuda, duration_5090_vk_id, duration_5090_vk_pip, b_5090_vk_IP, "RTX 5090");

		let numVisibleTriangles = b_5090_cuda["#visibleTriangles"];
		let strTris = numVisibleTriangles.toFixed(0);
		if(numVisibleTriangles > 1_000) strTris = (numVisibleTriangles / 1_000).toFixed(1) + "k";
		if(numVisibleTriangles > 1_000_000) strTris = (numVisibleTriangles / 1_000_000).toFixed(1) + "M";
		if(numVisibleTriangles > 1000_000_000) strTris = (numVisibleTriangles / 1_000_000_000).toFixed(1) + "B";
		strTris = strTris.padStart(7);

		if(r === rows.length - 1){
			console.log(`\t\\multirow{-13}{*}{\\rotatebox{90}{${view}}} `);
		}

		let line = `\t& ${items[0].scene.padEnd(20)} & ${strTris} & ${flags_c} & ${flags_m} & ${flags_j} & ${flags_h} `;
		line += `& ${str_4070_cuda} & ${str_4070_vk_id} & ${str_4070_vk_pip} `;
		line += `& ${str_4090_cuda} & ${str_4090_vk_id} & ${str_4090_vk_pip} `;
		line += `& ${str_5090_cuda} & ${str_5090_vk_id} & ${str_5090_vk_pip} \\\\`;
		console.log(line);

		if(r === rows.length - 1 && view === "closeup"){
			console.log("");
			console.log("\t\\midrule");
			console.log("");
		}

	}
}

console.log(`
	\\bottomrule
\\end{tabularx}
\\end{table*}
`);