
import * as fs from "fs";
import * as path from "path";


let inputDir = "E:/temp/jpeg_test"
// let outPath = "E:/temp/jpeg_test/results.html"

//csvPath = "E:/temp/jpeg_test/eval/statue.png/data.csv"
// let paths = [
// 	"E:/temp/jpeg_test/eval/bricks.png",
// 	"E:/temp/jpeg_test/eval/bridge.png",
// 	"E:/temp/jpeg_test/eval/bridge_1024_768.png",
// 	"E:/temp/jpeg_test/eval/coast_sand.png",
// 	"E:/temp/jpeg_test/eval/coast_sand_01_diff_4k.png",
// 	"E:/temp/jpeg_test/eval/coral_gravel_diff_4k.png",
// 	"E:/temp/jpeg_test/eval/fabric.png",
// 	"E:/temp/jpeg_test/eval/flowers.png",
// 	"E:/temp/jpeg_test/eval/graffiti.png",
// 	"E:/temp/jpeg_test/eval/Ground082S_4K-PNG_Color.png",
// 	"E:/temp/jpeg_test/eval/hill.png",
// 	"E:/temp/jpeg_test/eval/house.png",
// 	"E:/temp/jpeg_test/eval/house2.png",
// 	"E:/temp/jpeg_test/eval/kyumizudera.png",
// 	"E:/temp/jpeg_test/eval/Onyx010_4K-PNG_Color.png",
// 	"E:/temp/jpeg_test/eval/PavingStones126A_4K-PNG_Color.png",
// 	"E:/temp/jpeg_test/eval/PavingStones138_4K-PNG_Color.png",
// 	"E:/temp/jpeg_test/eval/plastic.png",
// 	"E:/temp/jpeg_test/eval/river.png",
// 	"E:/temp/jpeg_test/eval/snow.png",
// 	"E:/temp/jpeg_test/eval/statue.png",
// 	"E:/temp/jpeg_test/eval/wall.png",
// 	"E:/temp/jpeg_test/eval/wall_1024_768.png",
// ];

let paths = [];
const files = fs.readdirSync(`${inputDir}/eval`);
files.forEach(file => {
	paths.push(`${inputDir}/eval/${file}`);
});




let strEntries = "";

for(let p of paths){

	let filename = path.basename(p);

	let strEntry = `	<!-- ROW -->
	<span class="grid-header first-col">${filename}</span>
	<div class="cell"><img src="./eval/${filename}/plot_psnr.png" /></div>
	<div class="cell"><img src="./eval/${filename}/plot_flip.png" /></div>
	<div class="cell"><img src="./eval/${filename}/plot_lpips.png" /></div>
	<div class="cell"><img src="./eval/${filename}/plot_ssim.png" /></div>
	<div class="cell"><img src="./converted/${filename}/jpegli_80.jpg" /></div>
	`;

	strEntries = strEntries + strEntry;

}

let str = `<html>
<head>
<style>
.image-grid {
	display: grid;
	grid-template-columns: 20px 1fr 1fr 1fr 1fr 0.5fr; 
	/* grid-template-columns: 20px 1fr 1fr 1fr 1fr 0.5fr;  */
	gap: 10px; /* space between images */
	width: 100%;
}

.image-grid img {
	width: 100%;   /* image fills its grid cell */
	height: auto;  /* keep aspect ratio */
	display: block;
}

.grid-header {
	font-weight: bold;
	justify-self: center;
	font-size:xx-large;
}
.label {
	/* justify-self: center; */
	font-weight: bold;
	font-size:xx-large;
	transform-origin: left top;
	transform: rotate(-90deg);
}
.first-col {
	writing-mode: vertical-rl;  /* vertical text */
	transform: rotate(180deg);  /* flip so it reads bottom-to-top */
	text-align: center;
}
.cell{
	align-self: center;
}
</style>
</head>
<body>

<div>
<div class="image-grid">
	<!-- HEADER -->
	<span class="grid-header"></span>
	<span class="grid-header">PSNR↑</span>
	<span class="grid-header">FLIP↓</span>
	<span class="grid-header">LPIPS↓</span>
	<span class="grid-header">SSIM↑</span>
	<span class="grid-header"></span>

	${strEntries}

</div>
</div>


</body>
</html>`;


let outPath = `${inputDir}/results.html`;
fs.writeFileSync(outPath, str);
