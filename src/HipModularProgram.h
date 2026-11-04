// HipModularProgram.h -- hiprtc-based runtime compilation for HIP/ROCm
// This replaces CudaModularProgram.h for AMD GPU support.
//
// Key difference from nvrtc+nvJitLink: HIP compiles directly to code objects
// (HSACO), there is no LTO IR intermediate or separate linking step.
// Multiple source files are combined into a single compilation unit.
//
// Copyright (c) 2026 Advanced Micro Devices, Inc.
// Author: Jeff Daily <jeff.daily@amd.com>

#pragma once

#include <string>
#include <unordered_map>
#include "compat_print.h"
#include <stacktrace>
#include <vector>
#include <mutex>
#include <functional>
#include <cmath>
#include <fstream>
#include <sstream>

#include "unsuck.hpp"
#include "Timer.h"

#include <hip/hip_runtime.h>
#include <hip/hip_ext.h>
#include <hip/hiprtc.h>

using std::string;
using std::vector;
using std::unordered_map;
using std::mutex;
using std::lock_guard;
using std::cout;
using std::endl;
using std::println;

#define HIPRTC_SAFE_CALL(x)                                               \
do {                                                                      \
    hiprtcResult result = x;                                              \
    if (result != HIPRTC_SUCCESS) {                                       \
        std::cerr << "\nerror: " #x " failed with error "                 \
                  << hiprtcGetErrorString(result) << '\n';                \
        exit(1);                                                          \
    }                                                                     \
} while(0)

struct OptionalLaunchSettings{
    uint32_t gridsize = 0;
    uint32_t blocksize = 0;
    vector<void*> args;
    bool measureDuration = false;
    hipStream_t stream;
};

// HipModule represents a single source file to be compiled
struct HipModule{

    static void hip_checked(hipError_t result){
        if(result != hipSuccess){
            cout << "HIP error code: " << result << endl;
        }
    };

    string path = "";
    string name = "";
    bool compiled = false;
    bool success = false;
    vector<string> defines;

    // HIP compiles directly to code object, no LTO IR
    size_t codeSize = 0;
    char* code = nullptr;

    HipModule(string path, string name){
        this->path = path;
        this->name = name;
    }

    ~HipModule(){
        if(code){
            delete[] code;
            code = nullptr;
        }
    }

    string getSource(){
        return readFile(path);
    }
};


struct HipModularProgram{

    struct HipModularProgramArgs{
        vector<string> modules;
        vector<string> kernels;
        vector<string> defines;
    };

    static void hip_checked(hipError_t result){
        if(result != hipSuccess){
            cout << "HIP error code: " << result << endl;
        }
    };

    vector<HipModule*> modules;
    unordered_map<string, hipDeviceptr_t> cachedGlobals;

    hipModule_t mod = nullptr;
    void* codeObject = nullptr;
    size_t codeSize = 0;

    vector<std::function<void(void)>> compileCallbacks;

    vector<string> kernelNames;
    unordered_map<string, hipFunction_t> kernels;
    vector<string> defines;

    unordered_map<string, hipEvent_t> events_launch_start;
    unordered_map<string, hipEvent_t> events_launch_end;

    int MAX_LAUNCH_DURATIONS = 50;
    unordered_map<string, vector<float>> last_launch_durations;
    unordered_map<string, int> launches_per_frame;

    inline static vector<HipModularProgram*> instances;
    mutex mtx_instances;

    HipModularProgram(){
        lock_guard<mutex> lock(mtx_instances);
        instances.push_back(this);
    }

    HipModularProgram(vector<string> modules){
        construct({.modules = modules,});

        lock_guard<mutex> lock(mtx_instances);
        instances.push_back(this);
    }

    HipModularProgram(HipModularProgramArgs args){
        this->defines = args.defines;

        construct(args);

        lock_guard<mutex> lock(mtx_instances);
        instances.push_back(this);
    }

    static HipModularProgram* fromCodeObject(void* code, int64_t size){
        HipModularProgram* program = new HipModularProgram();

        program->codeObject = malloc(size);
        memcpy(program->codeObject, code, size);
        program->codeSize = size;

        hip_checked(hipModuleLoadData(&program->mod, program->codeObject));

        { // Retrieve Kernels by enumeration (HIP >= 5.3)
            // Note: hipModuleGetFunctionCount/EnumerateFunctions may not be available
            // in all HIP versions. Use hipModuleGetFunction with known names as fallback.
        }

        return program;
    }

    void construct(HipModularProgramArgs args){
        vector<string> modulePaths = args.modules;

        for(auto modulePath : modulePaths){
            string moduleName = fs::path(modulePath).filename().string();
            HipModule* module = new HipModule(modulePath, moduleName);
            module->defines = args.defines;

            // Don't compile individual modules; we'll combine them
            monitorFile(modulePath, [&, module]() {
                compileAndLink();
            });

            modules.push_back(module);
        }

        compileAndLink();
    }

    // Combine all sources and compile to a single code object
    void compileAndLink(){
        auto tStart = now();

        cout << "================================================================================" << endl;
        cout << "=== HIP COMPILING" << endl;
        cout << "================================================================================" << endl;

        // Combine all source files into one compilation unit
        std::stringstream combined;

        // Add common headers and defines
        combined << "// Combined source for hiprtc compilation\n";
        combined << "#define __HIP_PLATFORM_AMD__ 1\n";
        combined << "#define USE_HIP 1\n";
#ifdef _WIN32
        // On Windows, MSVC STL <cmath> is pulled in by GLM. Under hiprtc, all
        // unannotated functions default to __device__, so MSVC cmath's fma/lerp
        // helper templates instantiate with both a host-side and a device-side fma
        // visible, causing "ambiguous call" errors.  Fix: suppress MSVC <cmath> via
        // its include guard (_CMATH_) and provide the std:: math names that GLM
        // needs by injecting `using` declarations from the global namespace, which
        // hiprtc populates with __device__ versions through __clang_hip_cmath.h.
        combined << "#define _CMATH_\n";
        // Inject std:: math aliases so GLM's std::sqrt/std::log/etc. compile.
        combined << "namespace std {\n";
        combined << "  using ::sqrt; using ::sqrtf;\n";
        combined << "  using ::log; using ::logf; using ::log2; using ::log2f;\n";
        combined << "  using ::exp; using ::expf; using ::exp2; using ::exp2f;\n";
        combined << "  using ::pow; using ::powf;\n";
        combined << "  using ::floor; using ::floorf;\n";
        combined << "  using ::ceil; using ::ceilf;\n";
        combined << "  using ::fabs; using ::fabsf;\n";
        combined << "  using ::fma; using ::fmaf;\n";
        combined << "  using ::fmod; using ::fmodf;\n";
        combined << "  using ::modf; using ::modff;\n";
        combined << "  using ::frexp; using ::frexpf;\n";
        combined << "  using ::ldexp; using ::ldexpf;\n";
        combined << "  using ::sin; using ::sinf;\n";
        combined << "  using ::cos; using ::cosf;\n";
        combined << "  using ::tan; using ::tanf;\n";
        combined << "  using ::atan; using ::atanf;\n";
        combined << "  using ::atan2; using ::atan2f;\n";
        combined << "  using ::asin; using ::asinf;\n";
        combined << "  using ::acos; using ::acosf;\n";
        combined << "  using ::abs;\n";
        combined << "}\n";
#endif

        // Get device architecture
        hipDevice_t device;
        hipDeviceGet(&device, 0);

        hipDeviceProp_t props;
        hipGetDeviceProperties(&props, device);
        string archName = props.gcnArchName;

        // Add architecture-specific defines
        combined << "#define __gfx" << archName.substr(3) << "__ 1\n";

        // Add user defines
        for(const string& define : defines){
            combined << "#define " << define << "\n";
        }
        combined << "\n";

        // Read and concatenate all source files
        for(auto module : modules){
            cout << "Including: " << module->path << endl;
            string source = readFile(module->path);

            // Add source with file markers for debugging
            combined << "\n// === BEGIN " << module->name << " ===\n";
            combined << source;
            combined << "\n// === END " << module->name << " ===\n";
        }

        string combinedSource = combined.str();

        // Get include directories from module paths
        vector<string> includeDirs;
        for(auto module : modules){
            string dir = fs::path(module->path).parent_path().string();
            if(std::find(includeDirs.begin(), includeDirs.end(), dir) == includeDirs.end()){
                includeDirs.push_back(dir);
            }
        }

        // Build compiler options
        vector<string> optStrings;
        optStrings.push_back("--std=c++20");
        // No -ffast-math: clang's fast-math implies -ffinite-math-only, which
        // breaks this pipeline's Infinity depth sentinels and NaN clear values
        // (nvcc --use_fast_math does not assume finite math). The compiler
        // warns "use of infinity is undefined behavior" under it.
        optStrings.push_back("-ferror-limit=0");

        // Add include paths (absolute: comgr does not reliably resolve relative
        // -I against the launching CWD, and some kernel headers include siblings
        // in other source dirs via a "./" path).
        for(const string& dir : includeDirs){
            optStrings.push_back("-I" + fs::absolute(dir).string());
        }
        optStrings.push_back("-I" + fs::absolute("./src").string());
        optStrings.push_back("-I" + fs::absolute(".").string());
        optStrings.push_back("-I" + fs::absolute("./include").string());
        optStrings.push_back("-I" + fs::absolute("./libs").string());

        // hiprtc does not search the clang resource-dir headers (stddef.h, etc.)
        // that the kernels pull in via <cmath>/<cooperative_groups.h>; add them.
        {
            vector<string> roots;
            if(const char* p = std::getenv("ROCM_PATH")) roots.push_back(p);
            if(const char* p = std::getenv("HIP_PATH"))  roots.push_back(p);
            roots.push_back("/opt/rocm");
            for(const string& root : roots){
                for(const string& sub : {string("/llvm/lib/clang"), string("/lib/llvm/lib/clang")}){
                    string clangDir = root + sub;
                    std::error_code ec;
                    if(!fs::exists(clangDir, ec)) continue;
                    for(auto& e : fs::directory_iterator(clangDir, ec)){
                        string inc = e.path().string() + "/include";
                        if(fs::exists(inc + "/stddef.h")) optStrings.push_back("-isystem" + inc);
                    }
                }
                std::error_code ecRoot;
                if(fs::exists(root + "/include/hip/hip_runtime.h", ecRoot))
                    optStrings.push_back("-isystem" + root + "/include");
            }
        }

        // GPU architecture
        string archOpt = "--gpu-architecture=" + archName;
        optStrings.push_back(archOpt);

        // No -fgpu-rdc: HIP cooperative launch does not require relocatable
        // device code, and with rdc hiprtcGetCode returns no loadable code
        // object (the bitcode would need a separate hiprtc link step).

        // Convert to C strings for hiprtc
        vector<const char*> opts;
        for(const string& opt : optStrings){
            opts.push_back(opt.c_str());
        }

        println("hiprtcCompileProgram arguments: ");
        for(auto opt : opts){
            println("    {}", opt);
        }
        println("======");

        // Create and compile the program
        hiprtcProgram prog;
        HIPRTC_SAFE_CALL(hiprtcCreateProgram(&prog,
                                             combinedSource.c_str(),
                                             "combined_kernels.hip",
                                             0, nullptr, nullptr));

        hiprtcResult compileResult = hiprtcCompileProgram(prog, opts.size(), opts.data());

        // Get compilation log
        size_t logSize;
        HIPRTC_SAFE_CALL(hiprtcGetProgramLogSize(prog, &logSize));
        if(logSize > 1){
            char* log = new char[logSize + 1];
            HIPRTC_SAFE_CALL(hiprtcGetProgramLog(prog, log));
            log[logSize] = 0;
            println("Compilation log: {}", log);
            delete[] log;
        }

        if(compileResult != HIPRTC_SUCCESS){
            println("ERROR: hiprtc compilation failed");
            hiprtcDestroyProgram(&prog);
            return;
        }

        // Get compiled code
        size_t codeObjSize;
        HIPRTC_SAFE_CALL(hiprtcGetCodeSize(prog, &codeObjSize));

        if(codeObject){
            free(codeObject);
        }
        codeObject = malloc(codeObjSize);
        codeSize = codeObjSize;
        HIPRTC_SAFE_CALL(hiprtcGetCode(prog, (char*)codeObject));

        hiprtcDestroyProgram(&prog);

        cout << std::format("compiled code object. size: {} bytes\n", codeSize);

        // Drain the default stream before loading the module: hipModuleLoadData
        // can block indefinitely if GPU work (e.g. kernel_clearFramebuffer) is
        // still queued or running on the default stream.
        hipDeviceSynchronize();

        // Load the module
        if(mod){
            hipModuleUnload(mod);
            mod = nullptr;
        }

        hipError_t loadResult = hipModuleLoadData(&mod, codeObject);
        if(loadResult != hipSuccess){
            println("ERROR: hipModuleLoadData failed with error {}", (int)loadResult);
            return;
        }

        // Clear cached state
        kernelNames.clear();
        kernels.clear();
        cachedGlobals.clear();

        // Note: HIP doesn't have hipModuleGetFunctionCount in all versions
        // Kernels will be looked up by name on first use via getKernel()

        for(auto& callback : compileCallbacks){
            callback();
        }

        printElapsedTime("HIP compile+link duration: ", tStart);
    }

    hipFunction_t getKernel(const string& name){
        if(kernels.find(name) == kernels.end()){
            hipFunction_t func;
            hipError_t result = hipModuleGetFunction(&func, mod, name.c_str());
            if(result != hipSuccess){
                println("ERROR: kernel '{}' not found in module", name);
                return nullptr;
            }
            kernels[name] = func;
            kernelNames.push_back(name);

            // Create timing events
            hipEvent_t event_start, event_end;
            hipEventCreate(&event_start);
            hipEventCreate(&event_end);
            events_launch_start[name] = event_start;
            events_launch_end[name] = event_end;
        }
        return kernels[name];
    }

    hipDeviceptr_t getGlobalsPointer(string name){
        if(!cachedGlobals.contains(name)){
            hipDeviceptr_t dptr = 0;
            size_t bytes = 0;
            auto result = hipModuleGetGlobal(&dptr, &bytes, mod, name.c_str());

            if(result == hipSuccess){
                cachedGlobals[name] = dptr;
            }else{
                println("ERROR: global variable '{}' not found in module (err {})", name, (int)result);
                return 0;
            }
        }

        return cachedGlobals[name];
    }

    void onCompile(std::function<void(void)> callback){
        compileCallbacks.push_back(callback);
    }

    void addLaunchDuration(string kernelName, float duration){
        last_launch_durations[kernelName].resize(MAX_LAUNCH_DURATIONS);
        last_launch_durations[kernelName][0] += duration;
        launches_per_frame[kernelName]++;
    }

    // ROCm's hipModuleLaunchKernel can consume kernelParams after the API call
    // returns (deferred packet submission), unlike CUDA which copies them
    // synchronously. Keep each launch's parameter array alive in a ring so the
    // runtime never reads a freed vector or popped stack frame.
    static void** keepAliveArgs(void** args, size_t n){
        static std::deque<std::vector<void*>> ring;
        static mutex ringMtx;
        lock_guard<mutex> lock(ringMtx);
        ring.emplace_back(args, args + n);
        if(ring.size() > 4096) ring.pop_front();
        return ring.back().data();
    }

    // NOTE: on ROCm 7.2 / gfx90a, kernels of the runtime-compiled modules
    // dispatched through launch()/launch2D intermittently raise GPU memory
    // faults, while identical dispatch sequences inlined at the call site are
    // reliable (kernel arguments, timing instrumentation, launch API and
    // module identity were all ruled out; cause not yet isolated). The
    // performance-critical call sites in CuRast_render.h therefore dispatch
    // inline via INLINE_LAUNCH_1D/2D.
    void launch(string kernelName, vector<void*> args, OptionalLaunchSettings launchArgs = {}){
        if(args.empty()) return;
        this->launch(kernelName, keepAliveArgs(args.data(), args.size()), launchArgs);
    }

    void launch(string kernelName, void** args, OptionalLaunchSettings launchArgs){
        auto custart = Timer::recordCudaTimestamp();

        hipFunction_t func = getKernel(kernelName);
        if(!func) return;

        auto res_launch = hipModuleLaunchKernel(func,
            launchArgs.gridsize, 1, 1,
            launchArgs.blocksize, 1, 1,
            0, launchArgs.stream, args, nullptr);

        if (res_launch != hipSuccess) {
            const char* str = hipGetErrorString(res_launch);
            printf("error: %s \n", str);
            cout << __FILE__ << " - " << __LINE__ << endl;
            println("kernel: {}", kernelName);
        }

        Timer::recordDuration(kernelName, custart, Timer::recordCudaTimestamp());
    }

    void launch(string kernelName, vector<void*> args, int count, hipStream_t stream = 0){
        if(count == 0 || args.empty()) return;
        this->launch(kernelName, keepAliveArgs(args.data(), args.size()), count, stream);
    }

    void launch(string kernelName, void** args, int count, hipStream_t stream = 0){
        if (count == 0) return;

        uint32_t blockSize = 256;
        uint32_t gridSize = (count + blockSize - 1) / blockSize;

        auto custart = Timer::recordCudaTimestamp();

        hipFunction_t func = getKernel(kernelName);
        if(!func) return;

        auto res_launch = hipModuleLaunchKernel(func,
            gridSize, 1, 1,
            blockSize, 1, 1,
            0, stream, args, nullptr);

        if(res_launch != hipSuccess){
            hip_checked(res_launch);
        }

        Timer::recordDuration(kernelName, custart, Timer::recordCudaTimestamp());
    }

    void launch2D(string kernelName, void** args, int width, int height, hipStream_t stream = 0){
        if (width == 0 || height == 0) return;

        uint32_t blockSize = 8;
        uint32_t gridSizeX = (width + blockSize - 1) / blockSize;
        uint32_t gridSizeY = (height + blockSize - 1) / blockSize;

        auto custart = Timer::recordCudaTimestamp();

        hipFunction_t func = getKernel(kernelName);
        if(!func) return;

        auto res_launch = hipModuleLaunchKernel(func,
            gridSizeX, gridSizeY, 1,
            blockSize, blockSize, 1,
            0, stream, args, nullptr);

        if (res_launch != hipSuccess) {
            const char* str = hipGetErrorString(res_launch);
            printf("error %d, %s \n", int(res_launch), str);
            println("{} - {}", __FILE__, __LINE__);
            println("failed to launch kernel \"{}\". gridSize: {} x {}", kernelName, gridSizeX, gridSizeY);
            std::cerr << to_string(std::stacktrace::current()) << std::endl;
            exit(42415);
        }

        Timer::recordDuration(kernelName, custart, Timer::recordCudaTimestamp());
    }

    // Launch a kernel (non-cooperatively) using occupancy-based grid size.
    // Use this for kernels whose while-loop work distribution only needs
    // block-level or warp-level sync (no grid.sync), but benefit from
    // launching at full occupancy so all SMs are busy.
    void launchOccupancyBased(string kernelName, vector<void*> args, OptionalLaunchSettings launchArgs = {}){
        if(args.empty()) return;
        this->launchOccupancyBased(kernelName, keepAliveArgs(args.data(), args.size()), launchArgs);
    }

    void launchOccupancyBased(string kernelName, void** args, OptionalLaunchSettings launchArgs = {}){
        auto custart = Timer::recordCudaTimestamp();

        hipDevice_t device;
        int numSMs;
        hipCtxGetDevice(&device);
        hipDeviceGetAttribute(&numSMs, hipDeviceAttributeMultiprocessorCount, device);

        int blockSize = launchArgs.blocksize > 0 ? launchArgs.blocksize : 128;

        hipFunction_t func = getKernel(kernelName);
        if(!func) return;

        int blocksPerSM;
        hipError_t resultcode = hipModuleOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerSM, func, blockSize, 0);
        if(resultcode != hipSuccess){
            hip_checked(resultcode);
            return;
        }

        int numBlocks = std::clamp(blocksPerSM * numSMs, 10, 100'000);

        auto res_launch = hipModuleLaunchKernel(func,
            numBlocks, 1, 1,
            blockSize, 1, 1,
            0, launchArgs.stream, args, nullptr);
        if(res_launch != hipSuccess){
            hip_checked(res_launch);
        }

        Timer::recordDuration(kernelName, custart, Timer::recordCudaTimestamp());
    }

    void launchCooperative(string kernelName, vector<void*> args, OptionalLaunchSettings launchArgs = {}){
        if(args.empty()) return;
        this->launchCooperative(kernelName, keepAliveArgs(args.data(), args.size()), launchArgs);
    }

    void launchCooperative(string kernelName, void** args, OptionalLaunchSettings launchArgs = {}){
        auto custart = Timer::recordCudaTimestamp();

        hipDevice_t device;
        int numSMs;
        hipCtxGetDevice(&device);
        hipDeviceGetAttribute(&numSMs, hipDeviceAttributeMultiprocessorCount, device);

        int blockSize = launchArgs.blocksize > 0 ? launchArgs.blocksize : 128;

        hipFunction_t func = getKernel(kernelName);
        if(!func) return;

        int numBlocks;
        hipError_t resultcode = hipModuleOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocks, func, blockSize, 0);
        if(resultcode != hipSuccess){
            hip_checked(resultcode);
            return;
        }

        numBlocks *= numSMs;
        numBlocks = std::clamp(numBlocks, 10, 100'000);

        auto res_launch = hipModuleLaunchCooperativeKernel(func,
            numBlocks, 1, 1,
            blockSize, 1, 1,
            0, launchArgs.stream, args);
        if(res_launch != hipSuccess){
            hip_checked(res_launch);
        }

        Timer::recordDuration(kernelName, custart, Timer::recordCudaTimestamp());
    }

    void clearTimings(){
        for(auto& [key, value] : last_launch_durations){
            for(size_t i = value.size() - 1; i > 0; i--){
                value[i] = value[i - 1];
            }
            value[0] = 0.0f;
        }

        for(auto& [key, value] : launches_per_frame){
            value = 0;
        }
    }

    float getAvgTiming(string kernelName){
        if(last_launch_durations.find(kernelName) != last_launch_durations.end()){
            float sum = 0.0f;
            for(float value : last_launch_durations[kernelName]){
                sum += value;
            }
            float avg = sum / float(last_launch_durations[kernelName].size());
            return avg;
        }else{
            return 0.0f;
        }
    }
};
