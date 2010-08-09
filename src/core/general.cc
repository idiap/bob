#include "general.h"

namespace Torch {

char msg[10000];

void fatalerror(const char *fmt, ...)
{
  va_list args;
  va_start(args, fmt);
  vsprintf(msg, fmt, args);
  printf("\n$ Fatal Error: %s\n\n", msg);
  fflush(stdout);
  va_end(args);
  exit(-1);
}

void error(const char *fmt, ...)
{
  va_list args;
  va_start(args, fmt);
  vsprintf(msg, fmt, args);
  printf("\n$ Error: %s\n\n", msg);
  fflush(stdout);
  va_end(args);
}

void warning(const char *fmt, ...)
{
  va_list args;
  va_start(args, fmt);
  vsprintf(msg, fmt, args);
  printf("! Warning: %s\n", msg);
  fflush(stdout);
  va_end(args);
}

void message(const char *fmt, ...)
{
  va_list args;
  va_start(args, fmt);
  vsprintf(msg, fmt, args);
  printf("# %s\n", msg);
  fflush(stdout);
  va_end(args);
}

void print(const char *fmt, ...)
{
  va_list args;
  va_start(args, fmt);
  vsprintf(msg, fmt, args);
  printf("%s", msg);
  fflush(stdout);
  va_end(args);
}

/// Functions used for sorting arrays with \c qsort
int compare_floats(const void* a, const void* b)
{
	const float* da = (const float*) a;
	const float* db = (const float*) b;

	return (*da > *db) - (*da < *db);
}

int compare_doubles(const void* a, const void* b)
{
	const double* da = (const double*) a;
	const double* db = (const double*) b;

	return (*da > *db) - (*da < *db);
}
}
