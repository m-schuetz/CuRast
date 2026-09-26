
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

inline void printElapsedTime(string label, double startTime) {

	double elapsed = now() - startTime;

	println("{}: {:.3f}s", label, elapsed);
}

// taken from: https://stackoverflow.com/questions/2602013/read-whole-ascii-file-into-c-stdstring/2602060
inline string readFile(string path) {

	std::ifstream t(path);
	std::string str;

	t.seekg(0, std::ios::end);
	str.reserve(t.tellg());
	t.seekg(0, std::ios::beg);

	str.assign((std::istreambuf_iterator<char>(t)),
		std::istreambuf_iterator<char>());

	return str;
}

struct EventQueue {

	static EventQueue* instance;
	vector<std::function<void()>> queue;
	mutex mtx;

	void add(std::function<void()> event) {
		mtx.lock();
		this->queue.push_back(event);
		mtx.unlock();
	}

	void process() {

		mtx.lock();
		vector<std::function<void()>> q = queue;
		queue = vector<std::function<void()>>();
		mtx.unlock();

		for (auto &event : q) {
			event();
		}
	}
};

inline EventQueue* EventQueue::instance = new EventQueue();

inline void schedule(std::function<void()> event) {
	EventQueue::instance->add(event);
}

inline void monitorFile(string file, std::function<void()> callback) {

	std::thread([file, callback]() {

		if (!fs::exists(file)) {
			cout << "ERROR(monitorFile): file does not exist: " << file << endl;

			return;
		}

		auto lastWriteTime = fs::last_write_time(fs::path(file));

		using namespace std::chrono_literals;

		while (true) {
			std::this_thread::sleep_for(20ms);

			auto currentWriteTime = fs::last_write_time(fs::path(file));

			if (currentWriteTime > lastWriteTime) {

				//callback();
				schedule(callback);

				lastWriteTime = currentWriteTime;
			}

		}

	}).detach();
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

// granularity is non-deduced so that e.g. uint64_t (unsigned long on linux) and 4llu can be mixed
template<typename T>
inline T roundUp(T number, std::type_identity_t<T> granularity){
	T count = (number + granularity - 1) / granularity;

	return count * granularity;
}

template<typename T>
size_t byteSizeOf(const vector<T>& v){
	return sizeof(T) * v.size();
}
