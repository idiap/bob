#ifndef MY_TIMER_INC
#define MY_TIMER_INC

#include "core/Object.h"

namespace Torch {

/** This class is designed to measure time in micro-seconds

    @author Sebastien Marcel (marcel@idiap.ch)
    @version 2.0
    \date
    @since 2.0
*/
class MTimer
{
	long long start_time;

public:
	/// hours
	int hours;

	/// minutes
	int minutes;

	/// seconds
	int seconds;

	/// milliseconds
	int mseconds;

	/// microseconds
	int useconds;

	/// nanoseconds
	int nseconds;

	/// create the timer
	MTimer();

	///
	virtual ~MTimer();

	/** reset the timer

	    The timer will count time starting from now, and the accumulated time is erased.
	*/
	void reset();

	/// stop the timer and return the total accumulated time
	long long stop();

	/// return the total accumulated time
	long long getRunTime();

	/// compute the time difference
	long long computeDelta(long long current_time_, long long previous_time_);
};

}

#endif
