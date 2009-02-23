#ifndef GENERAL_INC
#define GENERAL_INC

#define _USE_MATH_DEFINES

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <limits.h>
#include <stdarg.h>
#include <time.h>
#include <float.h>

// Old systems need that to define FLT_MAX and DBL_MAX
#ifndef DBL_MAX
#include <values.h>
#endif

namespace Torch {

//-----------------------------------

/// Print an error message. The program will exit.
void fatalerror(const char* fmt, ...);

/// Print an error message. The program will NOT exit.
void error(const char* fmt, ...);

/// Print a warning message.
void warning(const char* fmt, ...);

/// Print a message.
void message(const char* fmt, ...);

/// Like printf.
void print(const char* fmt, ...);

//-----------------------------------

#ifndef min
/// The min function
#define	min(a,b) ((a) > (b) ? (b) : (a))
#endif

#ifndef max
/// The max function
#define	max(a,b) ((a) > (b) ? (a) : (b))
#endif

#ifndef getInRange
/// The getInRange function (force some value to be in the [m, M] range)
#define	getInRange(v,m,M) ((v) < (m) ? (m) : ((v) > (M) ? (M) : (v)))
#endif

#ifndef isInRange
/// The isInRange function (checks if some value is in the [m, M] range)
#define	isInRange(v,m,M) ((v) >= (m) && (v) <= (M))
#endif

#ifndef isIndex
/// The isIndex function checks if some value is an index
#define	isIndex(v,N) ((v) >= 0 && (v) < (N))
#endif

#define IS_NEAR(var, value, delta) ((var >= (value - delta)) && (var <= (value + delta)))

#define FixI(v) (int) (v+0.5)

/// Macros to check for errors (fatal -> force exit, error -> return false)
#define CHECK_FATAL(expression)					\
{								\
	const bool condition = (expression);			\
	if (condition == false)					\
	{							\
		fatalerror("Error: in file [%s] at line [%d]!\n",	\
			__FILE__, __LINE__);			\
	}							\
}

#define CHECK_ERROR(expression)					\
{								\
	const bool condition = (expression);			\
	if (condition == false)					\
	{							\
		message("Error: in file [%s] at line [%d]!\n",	\
			__FILE__, __LINE__);			\
		return false;					\
	}							\
}

}

#endif
