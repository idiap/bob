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

#define IS_NEAR(var, value, delta) ((var >= (value - delta)) && (var <= (value + delta)))

#define FixI(v) (int) (v+0.5)

}

#endif
