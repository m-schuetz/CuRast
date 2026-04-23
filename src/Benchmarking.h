#pragma once

#include <string>
#include <vector>

#include "glm/glm.hpp"

using namespace std;
using glm::vec3;
using glm::ivec2;

struct Benchmarking{

	struct View {
		float yaw;
		float pitch;
		float radius;
		vec3 target;
	};

	struct Scenario{
		string path;
		string label               = "undefined";
		bool skipUVs               = false;
		bool skipNormals           = false;
		bool compress              = false;
		bool useJpegTextures       = false;
		// bool isMeshoptimized       = false;
		int imageDivisionFactor    = 1;
		DisplayAttribute attribute = DisplayAttribute::TEXTURE;
		mat4 transform             = mat4(1.0);
		ivec2 instances_count      = {1, 1};
		vec2 instances_spacing     = {3.0, 3.0};

		View view_closeup;
		View view_overview;

		function<bool(shared_ptr<SceneNode>)> filter = [](shared_ptr<SceneNode>){return true;};
	};

	static inline string datasetPath = "";
	static inline Scenario* request_scenario = nullptr;
	static inline Scenario* active_scenario = nullptr;
	static inline View active_view;
	// to start, set to e.g. 60. Will decrement each frame, measures ready at 0.
	static inline int measurementCountdown = -1;

	static inline bool isMeshoptimized(Scenario* scenario){
		return scenario->path.contains("_optimized");
	}

	static inline vector<Scenario> scenarios = {
		Scenario{
			.path      = "DATASETPATH/sponza-png_by_Ludicon.glb",
			.label     = "Sponza",
			.compress  = false,
			.transform = mat4(
				1.000,  0.000, 0.000, 0.000,
				0.000,  0.000, 1.000, 0.000,
				0.000, -1.000, 0.000, 0.000,
				0.000,  0.000, 0.000, 1.000),
			.view_closeup = {
				.yaw       = -4.731,
				.pitch     = 0.009,
				.radius    = 336.359,
				.target    = { -2.986, 32.881, 119.491},
			},
			.view_overview = {
				.yaw       = -6.288,
				.pitch     = -1.319,
				.radius    = 3644.351,
				.target    = { -105.171, 158.794, 167.036},
			},
		},
		Scenario{
			.path      = "DATASETPATH/hakone_1M.glb",
			.label     = "Lantern",
			.compress  = false,
			.transform = mat4(1.0),
			.view_closeup = {
				.yaw       = -14.528,
				.pitch     = 0.288,
				.radius    = 8.175,
				.target    = { 25.642, -19.092, 16.739, },
			},
			.view_overview = {
				.yaw       = -7.070,
				.pitch     = -0.515,
				.radius    = 37.564,
				.target    = { 25.607, -17.328, 8.340},
			},
		},
		Scenario{
			.path              = "DATASETPATH/hakone_1M.glb",
			.label             = "Lantern Instances",
			.compress          = false,
			.transform         = mat4(1.0),
			.instances_count   = {50, 60},
			.instances_spacing = {30.0, 30.0},
			.view_closeup = {
				.yaw       = -7.065,
				.pitch     = 0.125,
				.radius    = 34.149,
				.target    = { 41.702, -2.691, 20.533, },
			},
			.view_overview = {
				.yaw       = -7.070,
				.pitch     = -0.412,
				.radius    = 336.359,
				.target    = { 199.990, 150.663, -59.570, },
			},
		},
		Scenario{
			.path              = "DATASETPATH/komainu_kobe_60m.glb",
			.label             = "Komainu Kobe",
			.skipNormals       = false,
			.compress          = false,
			.useJpegTextures   = false,
			.transform         = mat4(1.0),
			.view_closeup = {
				.yaw       = -22.094,
				.pitch     = 0.499,
				.radius    = 0.269,
				.target    = { -0.038, -0.004, 4.150, },
			},
			.view_overview = {
				.yaw       = -8.654,
				.pitch     = -0.397,
				.radius    = 3.531,
				.target    = { 0.141, 0.183, 2.825, },
			},
		},
		Scenario{
			.path              = "DATASETPATH/komainu_kobe_60m_optimized.glb",
			.label             = "Komainu Kobe",
			.skipNormals       = false,
			.compress          = false,
			.useJpegTextures   = false,
			.transform         = mat4(1.0),
			.view_closeup = {
				.yaw       = -22.094,
				.pitch     = 0.499,
				.radius    = 0.269,
				.target    = { -0.038, -0.004, 4.150, },
			},
			.view_overview = {
				.yaw       = -8.654,
				.pitch     = -0.397,
				.radius    = 3.531,
				.target    = { 0.141, 0.183, 2.825, },
			},
		},
		Scenario{
			.path              = "DATASETPATH/iconem/VeniceGeneral-Airborne-flyover-400M-12x16k-local-binply/venice.gltf",
			.label             = "Venice",
			.skipNormals       = true, // We're not benchmarking shading, and we need that extra GPU memory
			.compress          = false,
			.useJpegTextures   = true,
			// .isMeshoptimized   = false,
			.transform         = mat4(1.0),
			.view_closeup = {
				.yaw       = -24.381,
				.pitch     = -0.599,
				.radius    = 289.923,
				.target    = { -152.324, -130.043, -3.711, },
			},
			.view_overview = {
				.yaw       = -18.968,
				.pitch     = -0.769,
				.radius    = 4180.978,
				.target    = { 352.960, -1134.931, -462.529, },
			},
		},
		Scenario{
			.path              = "DATASETPATH/iconem/VeniceGeneral-Airborne-flyover-400M-12x16k-local-binply/venice_optimized.gltf",
			.label             = "Venice",
			.skipNormals       = true, // We're not benchmarking shading, and we need that extra GPU memory
			.compress          = false,
			.useJpegTextures   = true,
			.transform         = mat4(1.0),
			.view_closeup = {
				.yaw       = -24.381,
				.pitch     = -0.599,
				.radius    = 289.923,
				.target    = { -152.324, -130.043, -3.711, },
			},
			.view_overview = {
				.yaw       = -18.968,
				.pitch     = -0.769,
				.radius    = 4180.978,
				.target    = { 352.960, -1134.931, -462.529, },
			},
		},
		Scenario{
			.path                = "DATASETPATH/iconem/VeniceGeneral-Airborne-flyover-400M-12x16k-local-binply/venice.gltf",
			.label               = "Venice",
			.skipNormals         = true, // We're not benchmarking shading, and we need that extra GPU memory
			.compress            = false,
			.useJpegTextures     = false,
			.imageDivisionFactor = 2,
			.transform           = mat4(1.0),
			.view_closeup = {
				.yaw       = -24.381,
				.pitch     = -0.599,
				.radius    = 289.923,
				.target    = { -152.324, -130.043, -3.711, },
			},
			.view_overview = {
				.yaw       = -18.968,
				.pitch     = -0.769,
				.radius    = 4180.978,
				.target    = { 352.960, -1134.931, -462.529, },
			},
		},
		Scenario{
			.path                = "DATASETPATH/iconem/VeniceGeneral-Airborne-flyover-400M-12x16k-local-binply/venice_optimized.gltf",
			.label               = "Venice",
			.skipNormals         = true, // We're not benchmarking shading, and we need that extra GPU memory
			.compress            = false,
			.useJpegTextures     = false,
			.imageDivisionFactor = 2,
			.transform           = mat4(1.0),
			.view_closeup = {
				.yaw       = -24.381,
				.pitch     = -0.599,
				.radius    = 289.923,
				.target    = { -152.324, -130.043, -3.711, },
			},
			.view_overview = {
				.yaw       = -18.968,
				.pitch     = -0.769,
				.radius    = 4180.978,
				.target    = { 352.960, -1134.931, -462.529, },
			},
		},
		Scenario{
			.path                = "DATASETPATH/iconem/VeniceGeneral-Airborne-flyover-400M-12x16k-local-binply/venice.gltf",
			.label               = "Venice",
			.skipNormals         = true, // We're not benchmarking shading, and we need that extra GPU memory
			.compress            = true,
			.useJpegTextures     = false,
			.imageDivisionFactor = 2,
			.transform           = mat4(1.0),
			.view_closeup = {
				.yaw       = -24.381,
				.pitch     = -0.599,
				.radius    = 289.923,
				.target    = { -152.324, -130.043, -3.711, },
			},
			.view_overview = {
				.yaw       = -18.968,
				.pitch     = -0.769,
				.radius    = 4180.978,
				.target    = { 352.960, -1134.931, -462.529, },
			},
		},
		Scenario{
			.path                = "DATASETPATH/iconem/VeniceGeneral-Airborne-flyover-400M-12x16k-local-binply/venice_optimized.gltf",
			.label               = "Venice",
			.skipNormals         = true, // We're not benchmarking shading, and we need that extra GPU memory
			.compress            = true,
			.useJpegTextures     = false,
			.imageDivisionFactor = 2,
			.transform           = mat4(1.0),
			.view_closeup = {
				.yaw       = -24.381,
				.pitch     = -0.599,
				.radius    = 289.923,
				.target    = { -152.324, -130.043, -3.711, },
			},
			.view_overview = {
				.yaw       = -18.968,
				.pitch     = -0.769,
				.radius    = 4180.978,
				.target    = { 352.960, -1134.931, -462.529, },
			},
		},
		Scenario{
			.path              = "DATASETPATH/zorah_main_public.gltf/zorah_main_public.gltf",
			.label             = "Zorah",
			.skipUVs           = true,
			.compress          = true,
			.useJpegTextures   = false,
			.attribute         = DisplayAttribute::NONE,
			.transform         = mat4(
				1.000,  0.000, 0.000, 0.000,
				0.000,  0.000, 1.000, 0.000,
				0.000, -1.000, 0.000, 0.000,
				0.000,  0.000, 0.000, 1.000),
			.view_closeup = {
				.yaw       = -17.344,
				.pitch     = 0.073,
				.radius    = 10.893,
				.target    = { 44.294, 1.156, 6.458, },
			},
			.view_overview = {
				.yaw       = -16.537,
				.pitch     = -0.472,
				.radius    = 97.539,
				.target    = { 17.436, -6.343, 3.689, },
			},
			.filter = [](shared_ptr<SceneNode> node) -> bool {
				// Remove some billboards for aesthetic reasons
				if(node->name == "FogCard") return false;
				if(node->name == "Plane") return false;

				return true;
			},
		},
		Scenario{
			.path              = "DATASETPATH/zorah_main_public.gltf_optimized/zorah_main_public.gltf",
			.label             = "Zorah",
			.skipUVs           = true,
			.compress          = true,
			.useJpegTextures   = false,
			.attribute         = DisplayAttribute::NONE,
			.transform         = mat4(
				1.000,  0.000, 0.000, 0.000,
				0.000,  0.000, 1.000, 0.000,
				0.000, -1.000, 0.000, 0.000,
				0.000,  0.000, 0.000, 1.000),
			.view_closeup = {
				.yaw       = -17.344,
				.pitch     = 0.073,
				.radius    = 10.893,
				.target    = { 44.294, 1.156, 6.458, },
			},
			.view_overview = {
				.yaw       = -16.537,
				.pitch     = -0.472,
				.radius    = 97.539,
				.target    = { 17.436, -6.343, 3.689, },
			},
			.filter = [](shared_ptr<SceneNode> node) -> bool {
				// Remove some billboards for aesthetic reasons
				if(node->name == "FogCard") return false;
				if(node->name == "Plane") return false;

				return true;
			},
		},
	};

};






