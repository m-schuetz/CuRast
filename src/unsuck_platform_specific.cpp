#include "unsuck.hpp"

EventQueue *EventQueue::instance = new EventQueue();

#ifdef _WIN32
	#include "TCHAR.h"
	#include "pdh.h"
	#include "windows.h"
	#include "psapi.h"

void toClipboard(string str) {
	const char* output = str.c_str();
	const size_t len = strlen(output) + 1;
	HGLOBAL hMem = GlobalAlloc(GMEM_MOVEABLE, len);
	memcpy(GlobalLock(hMem), output, len);
	GlobalUnlock(hMem);
	OpenClipboard(0);
	EmptyClipboard();
	SetClipboardData(CF_TEXT, hMem);
	CloseClipboard();
}

uint64_t getPhysicalSectorSize(string path) {

	string canonicalPath = fs::canonical(fs::absolute(path)).string();

	std::string rootPath = fs::path(canonicalPath).root_path().string();
	rootPath = rootPath.substr(0, rootPath.find_last_of("\\"));
	rootPath = rootPath.substr(0, rootPath.find_last_of("/"));

#ifdef __cpp_lib_format
	std::string strDisk = format("\\\\.\\{}", rootPath);
#else
  std::string strDisk = format("\\\\.\\{}", rootPath);
#endif

	LPCSTR lpcstrDisk = strDisk.c_str();
	HANDLE hDevice = CreateFile(
		lpcstrDisk,
		0, 0, NULL,
		OPEN_EXISTING, 0, NULL
	);

	DWORD outsize;
	STORAGE_PROPERTY_QUERY storageQuery = {
		.PropertyId = StorageAccessAlignmentProperty,
		.QueryType = PropertyStandardQuery,
	};
	STORAGE_ACCESS_ALIGNMENT_DESCRIPTOR diskAlignment = { 0 };

	DeviceIoControl(hDevice,
		IOCTL_STORAGE_QUERY_PROPERTY,
		&storageQuery,
		sizeof(STORAGE_PROPERTY_QUERY),
		&diskAlignment,
		sizeof(STORAGE_ACCESS_ALIGNMENT_DESCRIPTOR),
		&outsize,
		NULL
	);

	return diskAlignment.BytesPerPhysicalSector;
}

void hideConsole(){
	ShowWindow(GetConsoleWindow(), SW_HIDE); 
}

// see https://stackoverflow.com/questions/63166/how-to-determine-cpu-and-memory-consumption-from-inside-a-process
MemoryData getMemoryData() {

	MemoryData data;

	{
		MEMORYSTATUSEX memInfo;
		memInfo.dwLength = sizeof(MEMORYSTATUSEX);
		GlobalMemoryStatusEx(&memInfo);
		DWORDLONG totalVirtualMem = memInfo.ullTotalPageFile;
		DWORDLONG virtualMemUsed = memInfo.ullTotalPageFile - memInfo.ullAvailPageFile;;
		DWORDLONG totalPhysMem = memInfo.ullTotalPhys;
		DWORDLONG physMemUsed = memInfo.ullTotalPhys - memInfo.ullAvailPhys;

		data.virtual_total = totalVirtualMem;
		data.virtual_used = virtualMemUsed;

		data.physical_total = totalPhysMem;
		data.physical_used = physMemUsed;

	}

	{
		PROCESS_MEMORY_COUNTERS_EX pmc;
		GetProcessMemoryInfo(GetCurrentProcess(), (PROCESS_MEMORY_COUNTERS*)&pmc, sizeof(pmc));
		SIZE_T virtualMemUsedByMe = pmc.PrivateUsage;
		SIZE_T physMemUsedByMe = pmc.WorkingSetSize;

		static size_t virtualUsedMax = 0;
		static size_t physicalUsedMax = 0;

		virtualUsedMax = std::max(virtualMemUsedByMe, virtualUsedMax);
		physicalUsedMax = std::max(physMemUsedByMe, physicalUsedMax);

		data.virtual_usedByProcess = virtualMemUsedByMe;
		data.virtual_usedByProcess_max = virtualUsedMax;
		data.physical_usedByProcess = physMemUsedByMe;
		data.physical_usedByProcess_max = physicalUsedMax;
	}


	return data;
}


void printMemoryReport() {

	auto memoryData = getMemoryData();
	double vm = double(memoryData.virtual_usedByProcess) / (1024.0 * 1024.0 * 1024.0);
	double pm = double(memoryData.physical_usedByProcess) / (1024.0 * 1024.0 * 1024.0);

	stringstream ss;
	ss << "memory usage: "
		<< "virtual: " << formatNumber(vm, 1) << " GB, "
		<< "physical: " << formatNumber(pm, 1) << " GB"
		<< endl;

	cout << ss.str();

}

void launchMemoryChecker(int64_t maxMB, double checkInterval) {

	auto interval = std::chrono::milliseconds(int64_t(checkInterval * 1000));

	thread t([maxMB, interval]() {

		static double lastReport = 0.0;
		static double reportInterval = 1.0;
		static double lastUsage = 0.0;
		static double largestUsage = 0.0;

		while (true) {
			auto memdata = getMemoryData();

			using namespace std::chrono_literals;
			std::this_thread::sleep_for(interval);
		}

	});
	t.detach();

}

static ULARGE_INTEGER lastCPU, lastSysCPU, lastUserCPU;
static int numProcessors;
static HANDLE self;
static bool initialized = false;

void init() {
	SYSTEM_INFO sysInfo;
	FILETIME ftime, fsys, fuser;

	GetSystemInfo(&sysInfo);
	// numProcessors = sysInfo.dwNumberOfProcessors;
	numProcessors = std::thread::hardware_concurrency();

	GetSystemTimeAsFileTime(&ftime);
	memcpy(&lastCPU, &ftime, sizeof(FILETIME));

	self = GetCurrentProcess();
	GetProcessTimes(self, &ftime, &ftime, &fsys, &fuser);
	memcpy(&lastSysCPU, &fsys, sizeof(FILETIME));
	memcpy(&lastUserCPU, &fuser, sizeof(FILETIME));

	initialized = true;
}

CpuData getCpuData() {
	FILETIME ftime, fsys, fuser;
	ULARGE_INTEGER now, sys, user;
	double percent;

	if (!initialized) {
		init();
	}

	GetSystemTimeAsFileTime(&ftime);
	memcpy(&now, &ftime, sizeof(FILETIME));

	GetProcessTimes(self, &ftime, &ftime, &fsys, &fuser);
	memcpy(&sys, &fsys, sizeof(FILETIME));
	memcpy(&user, &fuser, sizeof(FILETIME));
	percent = (sys.QuadPart - lastSysCPU.QuadPart) +
		(user.QuadPart - lastUserCPU.QuadPart);
	percent /= (now.QuadPart - lastCPU.QuadPart);
	percent /= numProcessors;
	lastCPU = now;
	lastUserCPU = user;
	lastSysCPU = sys;

	CpuData data;
	data.numProcessors = numProcessors;
	data.usage = percent * 100.0;

	return data;
}

std::string win32_error_string(DWORD err){
    LPSTR msg = nullptr;

    DWORD len = FormatMessageA(
        FORMAT_MESSAGE_ALLOCATE_BUFFER |
        FORMAT_MESSAGE_FROM_SYSTEM |
        FORMAT_MESSAGE_IGNORE_INSERTS,
        nullptr,
        err,
        MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT),
        (LPSTR)&msg,
        0,
        nullptr
    );

    std::string s = (len && msg) ? std::string(msg, len) : "Unknown error";

    if (msg) LocalFree(msg);

    return s;
}

void printLastError() {
	DWORD err = GetLastError();

	if (err != 0) {
		std::cout << "Encountered an Error. code=" << err
		<< " msg=" << win32_error_string(err) << "\n";
		//SetLastError(0);
	} else {
		println("no errors");
	}
	

}

shared_ptr<UnbufferedFile> UnbufferedFile::open(string path){

	shared_ptr<UnbufferedFile> file = make_shared<UnbufferedFile>();
	file->path = path;

	//printLastError();

	LPCSTR lpath = path.c_str();
	HANDLE hFile = CreateFileA(
		lpath,
		GENERIC_READ,
		FILE_SHARE_READ,
		nullptr,
		OPEN_EXISTING,
		FILE_FLAG_NO_BUFFERING | FILE_FLAG_OVERLAPPED,
		nullptr
	);

	//printLastError();

	file->sectorSize = getPhysicalSectorSize(path);
	file->handle = hFile;

	if (hFile == INVALID_HANDLE_VALUE) {
		println("ERROR: failed to CreateFileA");
		println("path:  {}", path);;
		println("{}", stacktrace::current());
		exit(6362345);
	}

	return file;
}

void UnbufferedFile::read(uint64_t start, uint64_t size, void* target){
	constexpr uint64_t CHUNK_SIZE = 2llu * 1024 * 1024 * 1024; // 2 GB

	uint64_t remaining = size;
	uint64_t offset    = start;
	uint8_t* dst       = reinterpret_cast<uint8_t*>(target);

	while (remaining > 0) {
		uint64_t chunkSize = std::min(remaining, CHUNK_SIZE);

		OVERLAPPED ov{};
		ov.Offset     = (offset >>  0llu) & 0xffffffffllu;
		ov.OffsetHigh = (offset >> 32llu) & 0xffffffffllu;
		ov.hEvent = CreateEventA(nullptr, TRUE, FALSE, nullptr);

		BOOL ok = ReadFile(
			handle,
			dst,
			(DWORD)chunkSize,
			nullptr,        // must be NULL for overlapped
			&ov
		);

		if (!ok) {
			DWORD err = GetLastError();
			if (err != ERROR_IO_PENDING) {
				CloseHandle(ov.hEvent);
				SetLastError(err);
				printLastError();
				println("{}", stacktrace::current());
				__debugbreak();
				exit(72463623476);
			}
		}

		DWORD br = 0;
		ok = GetOverlappedResult(handle, &ov, &br, TRUE);

		DWORD err = ok ? ERROR_SUCCESS : GetLastError();
		CloseHandle(ov.hEvent);

		if (!ok) {
			SetLastError(err);
			printLastError();
			println("{}", stacktrace::current());
			exit(8457346564);
		}

		offset    += chunkSize;
		dst       += chunkSize;
		remaining -= chunkSize;
	}
}

void UnbufferedFile::close(){
	CloseHandle(handle);
}

void readBinaryFileUnbuffered(string path, uint64_t start, uint64_t size, void* target){

	LPCSTR lpath = path.c_str();
	HANDLE hFile = CreateFileA(
		lpath,
		GENERIC_READ,
		FILE_SHARE_READ,
		nullptr,
		OPEN_EXISTING,
		FILE_FLAG_NO_BUFFERING | FILE_FLAG_OVERLAPPED,
		nullptr
	);

	if (hFile == INVALID_HANDLE_VALUE) {
		println("ERROR: failed to CreateFileA");
		println("path:  {}", path);
		println("start: {}", start);
		println("size:  {}", size);
		println("{}", stacktrace::current());
		exit(6362345);
	}

	// DWORD bytesPerSector = getPhysicalSectorSize(path);

	OVERLAPPED ov{};
	ov.Offset     = (start >>  0llu) & 0xffffffffllu;
	ov.OffsetHigh = (start >> 32llu) & 0xffffffffllu;

	DWORD bytesRead = 0;

	BOOL ok = ReadFile( 
		hFile,
		target,
		(DWORD)size,
		nullptr,        // must be NULL for overlapped
		&ov
	);

	if (!ok) {
		if (GetLastError() == ERROR_IO_PENDING) {
			// wait for completion
			GetOverlappedResult(hFile, &ov, &bytesRead, TRUE);
		} else {
			// error
			println("ERROR");
			println("{}", stacktrace::current());
			exit(74353);
		}
	}

	CloseHandle(hFile);
}

#elif defined(__linux__)

// see https://stackoverflow.com/questions/63166/how-to-determine-cpu-and-memory-consumption-from-inside-a-process

#include "sys/types.h"
#include "sys/sysinfo.h"

#include "stdlib.h"
#include "stdio.h"
#include "string.h"

int parseLine(char* line){
    // This assumes that a digit will be found and the line ends in " Kb".
    int i = strlen(line);
    const char* p = line;
    
	while (*p < '0' || *p > '9'){ 
		p++;
	}
	
    line[i - 3] = '\0';
    i = atoi(p);
	
    return i;
}

int64_t getVirtualMemoryUsedByProcess(){ //Note: this value is in KB!
    FILE* file = fopen("/proc/self/status", "r");
    int64_t result = -1;
    char line[128];

    while (fgets(line, 128, file) != NULL){
        if (strncmp(line, "VmSize:", 7) == 0){
            result = parseLine(line);
            break;
        }
    }
    fclose(file);
	
	result = result * 1024;
	
    return result;
}

int64_t getPhysicalMemoryUsedByProcess(){ //Note: this value is in KB!
    FILE* file = fopen("/proc/self/status", "r");
    int64_t result = -1;
    char line[128];

    while (fgets(line, 128, file) != NULL){
        if (strncmp(line, "VmRSS:", 6) == 0){
            result = parseLine(line);
            break;
        }
    }
    fclose(file);
	
	result = result * 1024;
	
    return result;
}


MemoryData getMemoryData() {
	
	struct sysinfo memInfo;

	sysinfo (&memInfo);
	int64_t totalVirtualMem = memInfo.totalram;
	totalVirtualMem += memInfo.totalswap;
	totalVirtualMem *= memInfo.mem_unit;

	int64_t virtualMemUsed = memInfo.totalram - memInfo.freeram;
	virtualMemUsed += memInfo.totalswap - memInfo.freeswap;
	virtualMemUsed *= memInfo.mem_unit;
	
	int64_t totalPhysMem = memInfo.totalram;
	totalPhysMem *= memInfo.mem_unit;
	
	long long physMemUsed = memInfo.totalram - memInfo.freeram;
	physMemUsed *= memInfo.mem_unit;

	int64_t virtualMemUsedByMe = getVirtualMemoryUsedByProcess();
	int64_t physMemUsedByMe = getPhysicalMemoryUsedByProcess();


	MemoryData data;
	
	static int64_t virtualUsedMax = 0;
	static int64_t physicalUsedMax = 0;

	virtualUsedMax = std::max(virtualMemUsedByMe, virtualUsedMax);
	physicalUsedMax = std::max(physMemUsedByMe, physicalUsedMax);

	{
		data.virtual_total = totalVirtualMem;
		data.virtual_used = virtualMemUsed;
		data.physical_total = totalPhysMem;
		data.physical_used = physMemUsed;

	}

	{
		data.virtual_usedByProcess = virtualMemUsedByMe;
		data.virtual_usedByProcess_max = virtualUsedMax;
		data.physical_usedByProcess = physMemUsedByMe;
		data.physical_usedByProcess_max = physicalUsedMax;
	}


	return data;
}


void printMemoryReport() {

	auto memoryData = getMemoryData();
	double vm = double(memoryData.virtual_usedByProcess) / (1024.0 * 1024.0 * 1024.0);
	double pm = double(memoryData.physical_usedByProcess) / (1024.0 * 1024.0 * 1024.0);

	stringstream ss;
	ss << "memory usage: "
		<< "virtual: " << formatNumber(vm, 1) << " GB, "
		<< "physical: " << formatNumber(pm, 1) << " GB"
		<< endl;

	cout << ss.str();

}

void launchMemoryChecker(int64_t maxMB, double checkInterval) {

	auto interval = std::chrono::milliseconds(int64_t(checkInterval * 1000));

	thread t([maxMB, interval]() {

		static double lastReport = 0.0;
		static double reportInterval = 1.0;
		static double lastUsage = 0.0;
		static double largestUsage = 0.0;

		while (true) {
			auto memdata = getMemoryData();

			using namespace std::chrono_literals;
			std::this_thread::sleep_for(interval);
		}

	});
	t.detach();

}

static int numProcessors;
static bool initialized = false;
static unsigned long long lastTotalUser, lastTotalUserLow, lastTotalSys, lastTotalIdle;

void init() {
	numProcessors = std::thread::hardware_concurrency();
	
	FILE* file = fopen("/proc/stat", "r");
    fscanf(file, "cpu %llu %llu %llu %llu", &lastTotalUser, &lastTotalUserLow, &lastTotalSys, &lastTotalIdle);
    fclose(file);

	initialized = true;
}

double getCpuUsage(){
    double percent;
    FILE* file;
    unsigned long long totalUser, totalUserLow, totalSys, totalIdle, total;

    file = fopen("/proc/stat", "r");
    fscanf(file, "cpu %llu %llu %llu %llu", &totalUser, &totalUserLow, &totalSys, &totalIdle);
    fclose(file);

    if (totalUser < lastTotalUser || totalUserLow < lastTotalUserLow ||
        totalSys < lastTotalSys || totalIdle < lastTotalIdle){
        //Overflow detection. Just skip this value.
        percent = -1.0;
    }else{
        total = (totalUser - lastTotalUser) 
			+ (totalUserLow - lastTotalUserLow) 
			+ (totalSys - lastTotalSys);
        percent = total;
        total += (totalIdle - lastTotalIdle);
        percent /= total;
        percent *= 100;
    }

    lastTotalUser = totalUser;
    lastTotalUserLow = totalUserLow;
    lastTotalSys = totalSys;
    lastTotalIdle = totalIdle;

    return percent;
}

CpuData getCpuData() {
	
	if (!initialized) {
		init();
	}

	CpuData data;
	data.numProcessors = numProcessors;
	data.usage = getCpuUsage();

	return data;
}


#endif