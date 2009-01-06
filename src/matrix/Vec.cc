#include "Vec.h"
#include "mx_low_level.h"

namespace Torch {

Vec::Vec(double * ptr_, int n_dim)
	:	m_shouldDeletePtr(false)
{
	ptr = ptr_;
	n = n_dim;
}

Vec::Vec(int n_dim)
	:	m_shouldDeletePtr(true)
{
	n = n_dim;
	ptr = new double[n];
}

void Vec::copy(Vec * vec, int start_i)
{
	if (vec == this)
		return;

	double *ptr_r = vec->ptr + start_i;
	double *ptr_w = ptr + start_i;
	for (int i = 0; i < n - start_i; i++)
		*ptr_w++ = *ptr_r++;
}

void Vec::zero()
{
	double *ptr_w = ptr;
	for (int i = 0; i < n; i++)
		*ptr_w++ = 0.;
}

double Vec::norm1(Vec * weights)
{
	double sum = 0.0;
	double *ptr_x = ptr;
	if (weights)
	{
		double *ptr_w = weights->ptr;
		for (int i = 0; i < n; i++)
			sum += *ptr_w++ * fabs(*ptr_x++);
	}
	else
	{
		for (int i = 0; i < n; i++)
			sum += fabs(*ptr_x++);
	}

	return sum;
}

double Vec::norm2(Vec * weights)
{
	double sum = 0.0;
	double *ptr_x = ptr;
	if (weights)
	{
		double *ptr_w = weights->ptr;
		for (int i = 0; i < n; i++)
		{
			double z = *ptr_x++;
			sum += *ptr_w++ * z * z;
		}
	}
	else
	{
		for (int i = 0; i < n; i++)
		{
			double z = *ptr_x++;
			sum += z * z;
		}
	}

	return sqrt(sum);
}

double Vec::normInf()
{
	double *ptr_x = ptr;
	double max_val = fabs(*ptr_x++);

	for (int i = 1; i < n; i++)
	{
		double z = fabs(*ptr_x);
		if (max_val < z)
			max_val = z;
		ptr_x++;
	}

	return max_val;
}

double Vec::iP(Vec * vec, int start_i)
{
	return (mxIp__(ptr + start_i, vec->ptr + start_i, n - start_i));
}

Vec *Vec::subVec(int dim1, int dim2)
{
	Vec *vec = new Vec(ptr + dim1, dim2 - dim1 + 1);

	return (vec);
}

Vec::~Vec()
{
	if (m_shouldDeletePtr)
	{
		delete[] ptr;
	}
}

}

