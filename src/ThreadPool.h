#pragma once

#include <iostream>
#include <vector>
#include <thread>
#include <queue>
#include <functional>
#include <mutex>
#include <condition_variable>
#include <atomic>

class ThreadPool {

public:
	int numThreads = 0;

	ThreadPool(size_t numThreads) 
		: stop(false), activeTasks(0) 
	{
		this->numThreads = numThreads;

		for (size_t threadIndex = 0; threadIndex < numThreads; ++threadIndex) {
			workers.emplace_back([this, threadIndex] {
				while (true) {
					std::function<void(int)> task;
					{
						std::unique_lock<std::mutex> lock(this->queueMutex);
						this->condition.wait(lock, [this] {
							return this->stop || !this->tasks.empty();
						});

						if (this->stop && this->tasks.empty()) return;

						task = std::move(this->tasks.front());
						this->tasks.pop();
					}
					
					// Execute task and decrement the counter
					task(threadIndex);
					
					// Atomically decrement and notify the wait() function
					if (--activeTasks == 0) {
						std::unique_lock<std::mutex> lock(waitMutex);
						waitCondition.notify_all();
					}
				}
			});
		}
	}

	void enqueue(std::function<void(int)> task) {
		{
			std::unique_lock<std::mutex> lock(queueMutex);
			tasks.push(std::move(task));
			activeTasks++; // Increment active task count
		}
		condition.notify_one();
	}

	// New: Block until all tasks are finished
	void wait() {
		std::unique_lock<std::mutex> lock(waitMutex);
		waitCondition.wait(lock, [this] {
			return activeTasks == 0;
		});
	}

	~ThreadPool() {
		{
			std::unique_lock<std::mutex> lock(queueMutex);
			stop = true;
		}
		condition.notify_all();
		for (std::thread &worker : workers) {
			worker.join();
		}
	}

private:
	std::vector<std::thread> workers;
	std::queue<std::function<void(int)>> tasks;
	
	std::mutex queueMutex;
	std::condition_variable condition;
	
	// For the wait() functionality
	std::mutex waitMutex;
	std::condition_variable waitCondition;
	std::atomic<size_t> activeTasks;
	
	bool stop;
};