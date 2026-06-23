#pragma once

#include <functional>
#include <vector>

#include "unsuck.hpp"

using namespace std;

enum class Easing{
	LINEAR,
	CUBIC,
};

struct Animation{
	double duration;
	double time;
	Easing easing;
	function<void(double)> callback;
};

struct Tween{

	inline static double time = now();
	inline static vector<Animation> animations;

	// applies the chosen easing function to a normalized time value u in [0, 1]
	inline static double ease(double u, Easing easing){
		switch(easing){
			case Easing::CUBIC:
				// smoothstep-style ease-in-out
				return u * u * (3.0 - 2.0 * u);
			case Easing::LINEAR:
			default:
				return u;
		}
	}

	// duration in seconds
	inline static void animate(double duration, function<void(double)> callback, Easing easing = Easing::LINEAR){
		Animation a;
		a.duration = duration;
		a.time = 0.0;
		a.easing = easing;
		a.callback = callback;

		animations.push_back(a);
	}

	inline static void update(){

		double newTime = now();
		double delta = newTime - time;

		// update queued animations
		vector<int> finishedAnimationIndices;
		for(int i = 0; i < animations.size(); i++){

			Animation& animation = animations[i];
			animation.time += delta;

			double u = animation.time / animation.duration;
			u = clamp(u, 0.0, 1.0);

			animation.callback(ease(u, animation.easing));

			if(u >= 1.0){
				finishedAnimationIndices.push_back(i);
			}
		}

		// remove animations that finished
		for(int i = finishedAnimationIndices.size() - 1; i >= 0; i--){
			animations.erase(animations.begin() + i);
		}

		time = newTime;
	}


};