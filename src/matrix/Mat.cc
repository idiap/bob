#include "Mat.h"
#include <math.h>

namespace Torch {

Mat::Mat(double ** ptr_, int n_rows, int n_cols)
	:	m_shouldDeleteBase(false),
		m_shouldDeletePtr(false)
{
	ptr = ptr_;
	m = n_rows;
	n = n_cols;
	base = 0;
}

Mat::Mat(double * ptr_, int n_rows, int n_cols)
	:	m_shouldDeleteBase(false),
		m_shouldDeletePtr(true)
{
	m = n_rows;
	n = n_cols;
	base = 0;
	ptr = new double*[m];
	for (int i = 0; i < m; i++)
		ptr[i] = ptr_ + i * n;
}

Mat::Mat(int n_rows, int n_cols)
	:	m_shouldDeleteBase(true),
		m_shouldDeletePtr(true)
{
	m = n_rows;
	n = n_cols;
	base = new double[m * n];
	ptr = new double*[m];
	for (int i = 0; i < m; i++)
		ptr[i] = base + i * n;
}

void Mat::copy(Mat * mat)
{
	if (mat == this)
		return;

	for (int i = 0; i < m; i++)
	{
		double *ptr_r = mat->ptr[i];
		double *ptr_w = ptr[i];
		for (int j = 0; j < n; j++)
			*ptr_w++ = *ptr_r++;
	}
}

void Mat::zero()
{
	for (int i = 0; i < m; i++)
	{
		double *ptr_w = ptr[i];
		for (int j = 0; j < n; j++)
			*ptr_w++ = 0.;
	}
}

double Mat::norm1()
{
	double max_val = 0.;
	for (int j = 0; j < n; j++)
	{
		double sum = 0.0;
		for (int i = 0; i < m; i++)
			sum += fabs(ptr[i][j]);

		if (max_val < sum)
			max_val = sum;
	}

	return max_val;
}

double Mat::normFrobenius()
{
	double sum = 0.;
	for (int i = 0; i < m; i++)
	{
		double *ptr_x = ptr[i];
		for (int j = 0; j < n; j++)
		{
			double z = *ptr_x++;
			sum += z * z;
		}
	}

	return sqrt(sum);
}

double Mat::normInf()
{
	double max_val = 0.;
	for (int i = 0; i < m; i++)
	{
		double sum = 0.0;
		double *ptr_x = ptr[i];
		for (int j = 0; j < n; j++)
			sum += fabs(*ptr_x++);

		if (max_val < sum)
			max_val = sum;
	}

	return max_val;
}

Vec *Mat::getRow(int row, Vec * vec)
{
	if (!vec)
		vec = new Vec(n);

	double *ptr_w = vec->ptr;
	double *ptr_r = ptr[row];
	for (int i = 0; i < n; i++)
		*ptr_w++ = *ptr_r++;

	return (vec);
}

Vec *Mat::getCol(int col, Vec * vec)
{
	if (!vec)
		vec = new Vec(m);

	double *ptr_w = vec->ptr;
	for (int i = 0; i < m; i++)
		*ptr_w++ = ptr[i][col];

	return (vec);
}

void Mat::setRow(int row, Vec * vec)
{
	double *ptr_w = ptr[row];
	double *ptr_r = vec->ptr;
	for (int i = 0; i < n; i++)
		*ptr_w++ = *ptr_r++;
}

void Mat::setCol(int col, Vec * vec)
{
	double *ptr_r = vec->ptr;
	for (int i = 0; i < m; i++)
		ptr[i][col] = *ptr_r++;
}

Mat *Mat::subMat(int row1, int col1, int row2, int col2)
{
	Mat *mat = new Mat(ptr, row2 - row1 + 1, col2 - col1 + 1);
	for (int i = row1; i <= row2; i++)
		mat->ptr[i - row1] = &ptr[i][col1];

	return (mat);
}

Mat::~Mat()
{
	if (m_shouldDeleteBase)
	{
		delete[] base;
	}
	if (m_shouldDeletePtr)
	{
		delete[] ptr;
	}
}

}
