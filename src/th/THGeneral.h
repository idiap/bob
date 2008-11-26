#ifndef TH_GENERAL_INC
#define TH_GENERAL_INC

#include <stdlib.h>
#include <stdio.h>
#include <stdarg.h>
#include <math.h>
#include <limits.h>
#include <float.h>
#include <time.h>


#ifdef __cplusplus
# define TH_EXTERNC extern "C"
#else
# define TH_EXTERNC extern
#endif

#ifdef WIN32
# ifdef TH_EXPORTS
#  define TH_API TH_EXTERNC __declspec(dllexport)
# else
#  define TH_API TH_EXTERNC __declspec(dllimport)
# endif
#else
# define TH_API TH_EXTERNC
#endif

#define THInf DBL_MAX

#if !defined(inline)
# define inline
#endif

#ifndef M_PI
# define M_PI 3.14159265358979323846
#endif

#ifdef _MSC_VER
TH_API double log1p(const double x);
#endif

TH_API void THError(const char *fmt, ...);
TH_API void THSetErrorHandler( void (*torchErrorHandlerFunction)(const char *msg) );
TH_API void THArgCheck(int condition, int argNumber, const char *msg);
TH_API void THSetArgCheckHandler( void (*torchArgCheckHandlerFunction)(int condition, int argNumber, const char *msg) );
TH_API void* THAlloc(long size);
TH_API void* THRealloc(void *ptr, long size);
TH_API void THFree(void *ptr);

#endif
