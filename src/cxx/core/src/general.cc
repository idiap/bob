/**
 * @file cxx/core/src/general.cc
 * @date Tue Jan 18 17:07:26 2011 +0100
 * @author André Anjos <andre.anjos@idiap.ch>
 *
 * Copyright (C) 2011 Idiap Reasearch Institute, Martigny, Switzerland
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, version 3 of the License.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */
#include "core/general.h"

namespace bob {

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

/// Functions used for sorting arrays with <qsort>
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
