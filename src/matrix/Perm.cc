#include "Perm.h"

namespace Torch {

Perm::Perm(int n_dim)
{
	n = n_dim;
	ptr = new int[n];
}

void Perm::copy(Perm * perm)
{
	int *ptr_r = perm->ptr;
	int *ptr_w = ptr;

	for (int i = 0; i < n; i++)
		*ptr_w++ = *ptr_r++;
}

Perm::~Perm()
{
	delete[] ptr;
}

}
