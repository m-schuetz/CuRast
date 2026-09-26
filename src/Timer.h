#pragma once

#include <vector>
#include <queue>
#include <string>

#include <cuda_runtime.h>



using namespace std;

struct Timer{

	struct Timestamp{
		cudaEvent_t cudaEvent = nullptr;
	};

	struct Recording{
		Timestamp start;
		Timestamp end;
		string label;
		double milliseconds;
	};

	inline static queue<Timestamp> pool;
	inline static bool enabled = false;
	inline static vector<Timestamp> timestamps;
	inline static vector<Recording> recordings;


	static void init(){
		static bool initialized = false;

		if(!initialized){

			int poolSize = 1000;

			for(int i = 0; i < poolSize; i++){

				Timestamp timestamp;

				cudaEventCreate(&timestamp.cudaEvent);

				pool.push(timestamp);
			}

			initialized = true;
		}
	}

	static Timestamp recordCudaTimestamp(){

		if(!enabled) return Timestamp();
		init();

		Timestamp timestamp = pool.front();
		pool.pop();

		cudaEventRecord(timestamp.cudaEvent, 0);

		timestamps.push_back(timestamp);

		return timestamp;
	}

	static void recordDuration(string label, Timestamp start, Timestamp end){

		if(!enabled) return;
		init();

		Recording recording;
		recording.label = label;
		recording.start = start;
		recording.end = end;

		recordings.push_back(recording);
	}

	// Evaluate all pending timestamp queries, then clear them and put them back into the pool
	static vector<Recording> resolve(){

		if(!enabled) return vector<Recording>();
		init();

		for(Recording& recording : recordings){
			cudaDeviceSynchronize();
			float duration;
			cudaEventElapsedTime(&duration, recording.start.cudaEvent, recording.end.cudaEvent);

			recording.milliseconds = duration;
		}

		for(Timestamp timestamp : timestamps){
			pool.push(timestamp);
		}

		vector<Recording> returnvalue = recordings;

		timestamps.clear();
		recordings.clear();

		return returnvalue;
	}

};
