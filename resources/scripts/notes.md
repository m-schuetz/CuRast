
# Evaluate

#### Convert images into various compressed forms; store results in "converted" folders

```
python convert_images.py -o E:/temp/jpeg_test/converted -i `
"E:/temp/jpeg_test/textures/coast_sand_01_diff_4k.png" `
"E:/temp/jpeg_test/textures/coral_gravel_diff_4k.png" `
"E:/temp/jpeg_test/textures/rock_face_03_diff_4k.png" `
...
```

#### Compare using various quality metrics, store results in "eval" folders

```
python compare_images.py -o "E:/temp/jpeg_test/eval" -i `
"E:/temp/jpeg_test/converted/coast_sand_01_diff_4k.png" `
"E:/temp/jpeg_test/converted/coral_gravel_diff_4k.png" `
"E:/temp/jpeg_test/converted/rock_face_03_diff_4k.png" `
...
```

### Generate a results.html page that shows all generated plots

```
node .\generate_results_page.mjs
```



# Scribbles








python convert_images.py -o E:/temp/jpeg_test/converted -i "E:/temp/jpeg_test/textures/graffiti_tex3.png"
python compare_images.py -o "E:/temp/jpeg_test/eval" -i "E:/temp/jpeg_test/converted/graffiti_tex3.png"


python convert_images.py -o E:/temp/jpeg_test/converted -i "E:/temp/jpeg_test/textures/bridge_1024_768.png"
python compare_images.py -o "E:/temp/jpeg_test/eval" -i "E:/temp/jpeg_test/converted/bridge_1024_768.png"

python convert_images.py -o E:/temp/jpeg_test/converted -i "E:/temp/jpeg_test/textures/Chip005_1K-PNG_Color.png.png"
python compare_images.py -o "E:/temp/jpeg_test/eval" -i "E:/temp/jpeg_test/converted/Chip005_1K-PNG_Color.png"


python convert_images.py -o E:/temp/jpeg_test/converted -i `
"E:/temp/jpeg_test/textures/anita_mui.png" `
"E:/temp/jpeg_test/textures/asphalt_04_diff_4k.png" `
"E:/temp/jpeg_test/textures/Bark012_1K-PNG_Color.png" `
"E:/temp/jpeg_test/textures/brick_wall_006_diff_4k.png" `
"E:/temp/jpeg_test/textures/bricks.png" `
"E:/temp/jpeg_test/textures/bridge.png" `
"E:/temp/jpeg_test/textures/bridge_1024_768.png" `
"E:/temp/jpeg_test/textures/Carpet015_1K-PNG_Color.png" `
"E:/temp/jpeg_test/textures/checkered_pavement_tiles_diff_4k.png" `
"E:/temp/jpeg_test/textures/Chip005_1K-PNG_Color.png" `
"E:/temp/jpeg_test/textures/coast_sand.png" `
"E:/temp/jpeg_test/textures/coast_sand_01_diff_4k.png" `
"E:/temp/jpeg_test/textures/coral_gravel_diff_4k.png" `
"E:/temp/jpeg_test/textures/denmin_fabric_02_diff_4k.png" `
"E:/temp/jpeg_test/textures/dry_riverbed_rock_diff_4k.png" `
"E:/temp/jpeg_test/textures/fabric.png" `
"E:/temp/jpeg_test/textures/fabric_leather_02_diff_4k.png" `
"E:/temp/jpeg_test/textures/fabric_pattern_07_col_1_4k.png" `
"E:/temp/jpeg_test/textures/flowers.png" `
"E:/temp/jpeg_test/textures/GlazedTerracotta002_1K-PNG_Color.png" `
"E:/temp/jpeg_test/textures/graffiti.png" `
"E:/temp/jpeg_test/textures/graffiti_tex3.png" `
"E:/temp/jpeg_test/textures/gray_rocks_diff_4k.png" `
"E:/temp/jpeg_test/textures/Ground082S_4K-PNG_Color.png" `
"E:/temp/jpeg_test/textures/hill.png" `
"E:/temp/jpeg_test/textures/house.png" `
"E:/temp/jpeg_test/textures/house2.png" `
"E:/temp/jpeg_test/textures/kiyomizu-dera.png" `
"E:/temp/jpeg_test/textures/LeafSet022_1K-PNG_Color.png" `
"E:/temp/jpeg_test/textures/leaves_forest_ground_diff_4k.png" `
"E:/temp/jpeg_test/textures/Marble016_1K-PNG_Color.png" `
"E:/temp/jpeg_test/textures/metal_grate_rusty_diff_4k.png" `
"E:/temp/jpeg_test/textures/Moss001_1K-PNG_Color.png" `
"E:/temp/jpeg_test/textures/mud_cracked_dry_03_diff_4k.png" `
"E:/temp/jpeg_test/textures/Onyx006_1K-PNG_Color.png" `
"E:/temp/jpeg_test/textures/PaintedMetal017_1K-PNG_Color.png" `
"E:/temp/jpeg_test/textures/patterned_paving_02_diff_4k.png" `
"E:/temp/jpeg_test/textures/PavingStones126A_4K-PNG_Color.png" `
"E:/temp/jpeg_test/textures/PavingStones138_4K-PNG_Color.png" `
"E:/temp/jpeg_test/textures/pine_bark_diff_4k.png" `
"E:/temp/jpeg_test/textures/plastic.png" `
"E:/temp/jpeg_test/textures/random_bricks_thick_diff_4k.png" `
"E:/temp/jpeg_test/textures/river.png" `
"E:/temp/jpeg_test/textures/Rock059_1K-PNG_Color.png" `
"E:/temp/jpeg_test/textures/rock_face_03_diff_4k.png" `
"E:/temp/jpeg_test/textures/Rocks022_1K-PNG_Color.png" `
"E:/temp/jpeg_test/textures/Rubber002_1K-PNG_Color.png" `
"E:/temp/jpeg_test/textures/rustic_stone_wall_02_diff_4k.png" `
"E:/temp/jpeg_test/textures/rusty_metal_03_diff_4k.png" `
"E:/temp/jpeg_test/textures/snow.png" `
"E:/temp/jpeg_test/textures/statue.png" `
"E:/temp/jpeg_test/textures/Terrazzo012_4K-PNG_Color.png" `
"E:/temp/jpeg_test/textures/Tiles035_1K-PNG_Color.png" `
"E:/temp/jpeg_test/textures/Tiles077_1K-PNG_Color.png" `
"E:/temp/jpeg_test/textures/Tiles093_1K-PNG_Color.png" `
"E:/temp/jpeg_test/textures/Tiles101_1K-PNG_Color.png" `
"E:/temp/jpeg_test/textures/wall.png" `
"E:/temp/jpeg_test/textures/wall_1024_768.png" `
"E:/temp/jpeg_test/textures/weathered_brown_planks_diff_4k.png" 



Import-Csv -Path "E:\temp\jpeg_test\eval\Chip005_1K-PNG_Color.png\data.csv" -Delimiter ';' | Out-GridView

python compare_images.py -o "E:/temp/jpeg_test/eval" -i "E:/temp/jpeg_test/converted/asphalt_04_diff_4k.png"

.\ffmpeg.exe -i "D:\dev\workspaces\jpeg\tmp\asphalt_04_diff_4k.png" -c:v libaom-av1 -still-picture 1 -pix_fmt yuv444p -crf 50 ffmpeg_crf50_asphalt_04_diff_4k.avif


python compare_images.py -o "E:/temp/jpeg_test/eval" -i 
"E:/temp/jpeg_test/converted/brick_wall_006_diff_4k.png" `
"E:/temp/jpeg_test/converted/coral_gravel_diff_4k.png" `
"E:/temp/jpeg_test/converted/denmin_fabric_02_diff_4k.png" `
"E:/temp/jpeg_test/converted/Chip005_1K-PNG_Color.png"

python compare_images.py -o "E:/temp/jpeg_test/eval" -i "E:/temp/jpeg_test/converted/Chip005_1K-PNG_Color.png"




## NVIDIA Texture Tools
C:\Program Files\NVIDIA Corporation\NVIDIA Texture Tools

## ASTCENC
https://developer.arm.com/documentation/102162/0430/About-Arm-ASTC-Encoder?lang=en
D:\software\astcenc
	-fast
	-medium
	-thorough
	-verythorough
	-exhaustive

.\nvcompress.exe -color -nomips -production -bc1 "E:\temp\jpeg_test\nvcompress\original.jpg" "E:\temp\jpeg_test\nvcompress\bc1_nvcompress.dds"
.\nvcompress.exe -color -nomips -production -bc7 "E:\temp\jpeg_test\nvcompress\original.jpg" "E:\temp\jpeg_test\nvcompress\bc7_nvcompress.dds"
.\nvcompress.exe -color -nomips -production -astc_ldr_4x4 "E:\temp\jpeg_test\nvcompress\original.jpg" "E:\temp\jpeg_test\nvcompress\astc_ldr_4x4.dds"
.\nvcompress.exe -color -nomips -production -astc_ldr_8x8 "E:\temp\jpeg_test\nvcompress\original.jpg" "E:\temp\jpeg_test\nvcompress\astc_ldr_8x8.dds"

.\nvdecompress.exe -format png "E:\temp\jpeg_test\nvcompress\bc1_nvcompress.dds" "E:\temp\jpeg_test\nvcompress\bc1_nvcompress.png"
.\nvdecompress.exe -format png "E:\temp\jpeg_test\nvcompress\bc7_nvcompress.dds" "E:\temp\jpeg_test\nvcompress\bc7_nvcompress.png"

.\astcenc-sse2.exe -cl "E:\temp\jpeg_test\nvcompress\original.jpg" "E:\temp\jpeg_test\nvcompress\arm_astc_ldr_8x8.astc" 8x8 -medium
.\astcenc-sse2.exe -cl "E:\temp\jpeg_test\nvcompress\original.jpg" "E:\temp\jpeg_test\nvcompress\arm_astc_ldr_4x4.astc" 4x4 -medium


.\astcenc-sse2.exe -dl "E:\temp\jpeg_test\nvcompress\arm_astc_ldr_4x4.astc" "E:\temp\jpeg_test\nvcompress\arm_astc_ldr_4x4.png"
.\astcenc-sse2.exe -dl "E:\temp\jpeg_test\nvcompress\arm_astc_ldr_8x8.astc" "E:\temp\jpeg_test\nvcompress\arm_astc_ldr_8x8.png"
