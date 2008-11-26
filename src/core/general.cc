#include "general.h"

namespace Torch {

char msg[10000];

void error(const char *fmt, ...)
{
  va_list args;
  va_start(args, fmt);
  vsprintf(msg, fmt, args);
  printf("\n$ Error: %s\n\n", msg);
  fflush(stdout);
  va_end(args);
  exit(-1);
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

}
