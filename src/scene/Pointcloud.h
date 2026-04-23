
struct Pointcloud{
	string name = "undefined";
	uint32_t numPoints = 0;
	bool isLoaded = false;
	bool isLoading = false;

	CUdeviceptr cptr_position = 0;
	CUdeviceptr cptr_color = 0;
};