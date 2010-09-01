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
# define TH_API extern "C"
#else
# define TH_API extern
#endif

TH_API void THError(const char *fmt, ...);
TH_API void THSetErrorHandler( void (*torchErrorHandlerFunction)(const char *msg) );
TH_API void THArgCheck(int condition, int argNumber, const char *msg);
TH_API void THSetArgCheckHandler( void (*torchArgCheckHandlerFunction)(int condition, int argNumber, const char *msg) );
TH_API void* THAlloc(long size);
TH_API void* THRealloc(void *ptr, long size);
TH_API void THFree(void *ptr);

#endif
