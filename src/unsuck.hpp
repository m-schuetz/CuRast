
#pragma once

#include <string>
#include <vector>
#include <fstream>
#include <sstream>
#include <chrono>
#include <iostream>
#include <filesystem>
#include <execution>
#include <limits>
#include <random>
#include <memory>
#include <algorithm>
#include <thread>
#include <cstdint>
#include <cstring>
#include <functional>
#include <mutex>
#include <print>
#include <stacktrace>
#include <type_traits>
#include <csignal>

#define __debugbreak() std::raise(SIGTRAP)

using std::cout;
using std::endl;
using std::to_string;
using std::string;
using std::function;
using std::vector;
using std::ifstream;
using std::ofstream;
using std::fstream;
using std::streamsize;
using std::stringstream;
using std::thread;
using std::jthread;
using std::ios;
using std::shared_ptr;
using std::make_shared;
using std::chrono::high_resolution_clock;
using std::mutex;
using std::println;
using std::stacktrace;

namespace fs = std::filesystem;

static long long unsuck_start_time = high_resolution_clock::now().time_since_epoch().count();

inline double now() {
	auto now = std::chrono::high_resolution_clock::now();
	long long nanosSinceStart = now.time_since_epoch().count() - unsuck_start_time;

	double secondsSinceStart = double(nanosSinceStart) / 1'000'000'000.0;

	return secondsSinceStart;
}

class punct_facet : public std::numpunct<char> {
protected:
	char do_decimal_point() const { return '.'; };
	char do_thousands_sep() const { return '\''; };
	string do_grouping() const { return "\3"; }
};

inline std::locale getSaneLocale(){
	return std::locale(std::cout.getloc(), new punct_facet);
}

template<typename T>
size_t byteSizeOf(const vector<T>& v){
	return sizeof(T) * v.size();
}
