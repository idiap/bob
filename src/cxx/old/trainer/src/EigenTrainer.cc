/**
 * @file cxx/old/trainer/src/EigenTrainer.cc
 * @date Wed Apr 6 14:49:40 2011 +0200
 * @author Andre Anjos <andre.anjos@idiap.ch>
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
#include "trainer/EigenTrainer.h"

namespace Torch {

extern "C" int sort_inc_int_double(const void *p1, const void *p2)
{
	int_double v1 = *((int_double*)p1);
	int_double v2 = *((int_double*)p2);

	if (v1.the_double > v2.the_double) return 1;
	if (v1.the_double < v2.the_double) return -1;

	return 0;
}

extern "C" int sort_dec_int_double(const void *p1, const void *p2)
{
	int_double v1 = *((int_double*)p1);
	int_double v2 = *((int_double*)p2);

	if (v1.the_double < v2.the_double) return 1;
	if (v1.the_double > v2.the_double) return -1;

	return 0;
}

}

