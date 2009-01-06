#include "Perm_operations.h"

namespace Torch {

/**********************************************************************
Note: A permutation is often interpreted as a matrix
		(i.e. a permutation matrix).
	A permutation px represents a permutation matrix P where
		P[i][j] == 1 if and only if px->ptr[i] == j
**********************************************************************/


/* px_inv -- invert permutation -- in situ
	-- taken from ACM Collected Algorithms #250 */
void mxPermInv(Perm * px, Perm * out)
{
	int i, j, k, *p;

	out->copy(px);

	p = out->ptr;
	for (int n = out->n - 1; n >= 0; n--)
	{
		i = p[n];
		if (i < 0)
			p[n] = -1 - i;
		else
		{
			if (i != n)
			{
				k = n;
				while (1)
				{
					j = p[i];
					p[i] = -1 - k;
					if (j == n)
					{
						p[n] = i;
						break;
					}
					k = i;
					i = j;
				}
			}
		}
	}
}

/* px_mlt -- permutation multiplication (composition) */
// not in-situ
void mxPermMulPerm(Perm * px1, Perm * px2, Perm * out)
{
	int size = px1->n;
	for (int i = 0; i < size; i++)
		out->ptr[i] = px1->ptr[px2->ptr[i]];
}

/* px_vec -- permute vector */
// can be in-situ
void mxPermVec(Perm * px, Vec * vec, Vec * out)
{
	int old_i, i, size, start;
	double tmp;

	size = px->n;
	if (size == 0)
	{
		out->copy(vec);
		return;
	}

	if (out != vec)
	{
		for (i = 0; i < size; i++)
			out->ptr[i] = vec->ptr[px->ptr[i]];
	}
	else
	{				/* in situ algorithm */
		start = 0;
		while (start < size)
		{
			old_i = start;
			i = px->ptr[old_i];
			if (i >= size)
			{
				start++;
				continue;
			}
			tmp = vec->ptr[start];
			while (1)
			{
				vec->ptr[old_i] = vec->ptr[i];
				px->ptr[old_i] = i + size;
				old_i = i;
				i = px->ptr[old_i];
				if (i >= size)
					break;
				if (i == start)
				{
					vec->ptr[old_i] = tmp;
					px->ptr[old_i] = i + size;
					break;
				}
			}
			start++;
		}

		for (i = 0; i < size; i++)
			px->ptr[i] = px->ptr[i] - size;
	}
}

/* pxinv_vec -- apply the inverse of px to x, returning the result in out */
// can be in-situ, but "oh booooy!"...
void mxPermInvVec(Perm * px, Vec * x, Vec * out)
{
	int size = px->n;
	if (size == 0)
	{
		out->copy(x);
		return;
	}
	if (out != x)
	{
		for (int i = 0; i < size; i++)
			out->ptr[px->ptr[i]] = x->ptr[i];
	}
	else
	{				/* in situ algorithm --- cheat's way out */
		mxPermInv(px, px);
		mxPermVec(px, x, out);
		mxPermInv(px, px);
	}
}



/* px_transp -- transpose elements of permutation
		-- Really multiplying a permutation by a transposition */
/* i1, i2 <---> elements to transpose */
void mxTrPerm(Perm * px, int i1, int i2)
{
	int temp;

	if (i1 < px->n && i2 < px->n)
	{
		temp = px->ptr[i1];
		px->ptr[i1] = px->ptr[i2];
		px->ptr[i2] = temp;
	}
}

/* myqsort -- a cheap implementation of Quicksort on integers
		-- returns number of swaps */
static int mxQsort(int *a, int num)
{
	int i, j, tmp, v;
	int numswaps;

	numswaps = 0;
	if (num <= 1)
		return 0;

	i = 0;
	j = num;
	v = a[0];
	for (;;)
	{
		while (a[++i] < v);
		while (a[--j] > v);
		if (i >= j)
			break;

		tmp = a[i];
		a[i] = a[j];
		a[j] = tmp;
		numswaps++;
	}

	tmp = a[0];
	a[0] = a[j];
	a[j] = tmp;
	if (j != 0)
		numswaps++;

	numswaps += mxQsort(&a[0], j);
	numswaps += mxQsort(&a[j + 1], num - (j + 1));

	return numswaps;
}


/* px_sign -- compute the ``sign'' of a permutation = +/-1 where
		px is the product of an even/odd # transpositions */
int mxPermSign(Perm * px)
{
	int numtransp;
	Perm *px2 = new Perm(px->n);

	px2->copy(px);
	numtransp = mxQsort(px2->ptr, px2->n);

	delete px2;

	return (numtransp % 2) ? -1 : 1;
}


/* px_cols -- permute columns of matrix A; out = A.px'
	-- May NOT be in situ */
void mxPermColsMat(Perm * px, Mat * mat, Mat * out)
{
	int m, n, px_j;
	double **mat_ptr, **out_ptr;

	m = mat->m;
	n = mat->n;
	mat_ptr = mat->ptr;
	out_ptr = out->ptr;

	for (int j = 0; j < n; j++)
	{
		px_j = px->ptr[j];
		for (int i = 0; i < m; i++)
			out_ptr[i][px_j] = mat_ptr[i][j];
	}
}

/* px_rows -- permute rows of matrix A; out = px.A
	-- May NOT be in situ */
void mxPermRowsMat(Perm * px, Mat * mat, Mat * out)
{
	int m, n, px_i;
	double **mat_ptr, **out_ptr;

	m = mat->m;
	n = mat->n;
	mat_ptr = mat->ptr;
	out_ptr = out->ptr;

	for (int i = 0; i < m; i++)
	{
		px_i = px->ptr[i];
		for (int j = 0; j < n; j++)
			out_ptr[i][j] = mat_ptr[px_i][j];
	}
}

}

