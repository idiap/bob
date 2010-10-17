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

