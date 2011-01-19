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
	int64_t start_time;

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
	int64_t stop();

	/// return the total accumulated time
	int64_t getRunTime();

	/// compute the time difference
	int64_t computeDelta(int64_t current_time_, int64_t previous_time_);
};

}

#endif
