import * as fs from "fs";
import * as njspath from "path";

let inputDir = "E:/temp/jpeg_test/converted"
let outPath = "E:/temp/jpeg_test/bpps.txt"

// clear file
fs.writeFileSync(outPath, "");

function listFiles(dir){
	let paths = [];
	const files = fs.readdirSync(dir);
	files.forEach(file => {
		paths.push(`${dir}/${file}`);
	});

	return paths;
}

let paths = listFiles(inputDir);

function getPngSize(path) {
	const fd = fs.openSync(path, "r");
	const buffer = Buffer.alloc(24);
	fs.readSync(fd, buffer, 0, 24, 0);
	fs.closeSync(fd);

	const width = buffer.readUInt32BE(16);
	const height = buffer.readUInt32BE(20);
	return { width, height };
}



for(let path of paths){
	if(path.endsWith(".png")){
		let filename = njspath.basename(path);
		let pathToPng = `${path}/${filename}`;
		let imageSize = getPngSize(pathToPng);

		
		console.log(`# ${filename.padEnd(40)}`);
		fs.appendFileSync(outPath, `# ${filename.padEnd(40)}\n`);


		let images = listFiles(path);

		for(let imagePath of images){
			let imageName = njspath.basename(imagePath);

			if(
				imageName.endsWith("astc_ldr_12x12.dds")
				|| imageName.startsWith("jpegli")
				|| imageName.startsWith("jpegXL")
				|| imageName.startsWith("AVIF")
			){
				let fileSize = fs.statSync(imagePath).size;
				let bpp = 8 * fileSize / (imageSize.width * imageSize.height);

				console.log(`    ${imageName.padEnd(25)}: ${bpp.toFixed(3)} bpp`);
				fs.appendFileSync(outPath, `    ${imageName.padEnd(25)}: ${bpp.toFixed(3)} bpp \n`);
			}
		}




	}else{
		console.log("ERROR: source is not a png file? path: ", path);
	}
}

// console.log(paths)