#include "mx_solve.h"
#include "mx_low_level.h"
#include "general.h"
//#include <math.h>

#ifndef INF
	#define INF	(10000000000.0)
#endif

namespace Torch {

/* Most matrix factorisation routines are in-situ unless otherwise specified */

/* Usolve -- back substitution with optional over-riding diagonal
		-- can be in-situ but doesn't need to be */
void mxUSolve(Mat * matrix, Vec * b, Vec * out, double diag)
{
	int dim;
	int i, i_lim;
	double **mat_ptr, *mat_row, *b_ptr, *out_ptr, *out_col, sum, tiny;

	dim = min(matrix->m, matrix->n);
	mat_ptr = matrix->ptr;
	b_ptr = b->ptr;
	out_ptr = out->ptr;

	tiny = 10.0 / INF;

	for (i = dim - 1; i >= 0; i--)
		if (b_ptr[i] != 0.0)
			break;
		else
			out_ptr[i] = 0.0;
	i_lim = i;

	for (; i >= 0; i--)
	{
		sum = b_ptr[i];
		mat_row = &mat_ptr[i][i + 1];
		out_col = &out_ptr[i + 1];
		sum -= mxIp__(mat_row, out_col, i_lim - i);
		if (diag == 0.0)
		{
			if (fabs(mat_ptr[i][i]) <= tiny * fabs(sum))
				error("USolve: sorry, singular problem.");
			else
				out_ptr[i] = sum / mat_ptr[i][i];
		}
		else
			out_ptr[i] = sum / diag;
	}
}

/* Lsolve -- forward elimination with (optional) default diagonal value */
void mxLSolve(Mat * matrix, Vec * b, Vec * out, double diag)
{
	int dim, i, i_lim;
	double **mat_ptr, *mat_row, *b_ptr, *out_ptr, *out_col, sum, tiny;

	dim = min(matrix->m, matrix->n);
	mat_ptr = matrix->ptr;
	b_ptr = b->ptr;
	out_ptr = out->ptr;

	for (i = 0; i < dim; i++)
		if (b_ptr[i] != 0.0)
			break;
		else
			out_ptr[i] = 0.0;
	i_lim = i;

	tiny = 10.0 / INF;

	for (; i < dim; i++)
	{
		sum = b_ptr[i];
		mat_row = &mat_ptr[i][i_lim];
		out_col = &out_ptr[i_lim];
		sum -= mxIp__(mat_row, out_col, i - i_lim);
		if (diag == 0.0)
		{
			if (fabs(mat_ptr[i][i]) <= tiny * fabs(sum))
				error("LSolve: sorry, singular problem.");
			else
				out_ptr[i] = sum / mat_ptr[i][i];
		}
		else
			out_ptr[i] = sum / diag;
	}
}


/* UTsolve -- forward elimination with (optional) default diagonal value
		using UPPER triangular part of matrix */
void mxUTSolve(Mat * mat, Vec * b, Vec * out, double diag)
{
	int dim, i, i_lim;
	double **mat_ptr, *b_ptr, *out_ptr, tmp, invdiag, tiny;

	dim = min(mat->m, mat->n);
	mat_ptr = mat->ptr;
	b_ptr = b->ptr;
	out_ptr = out->ptr;

	tiny = 10.0 / INF;

	for (i = 0; i < dim; i++)
	{
		if (b_ptr[i] != 0.0)
			break;
		else
			out_ptr[i] = 0.0;
	}
	i_lim = i;
	if (b != out)
	{
		mxZero__(out_ptr, out->n);
		double *ptr_r = &b_ptr[i_lim];
		double *ptr_w = &out_ptr[i_lim];
		for (int j = 0; j < dim - i_lim; j++)
			*ptr_w++ = *ptr_r++;
	}

	if (diag == 0.0)
	{
		for (; i < dim; i++)
		{
			tmp = mat_ptr[i][i];
			if (fabs(tmp) <= tiny * fabs(out_ptr[i]))
				error("UTSolve: sorry, singular problem.");
			out_ptr[i] /= tmp;
			mxDoubleMulAdd__(&out_ptr[i + 1], &mat_ptr[i][i + 1], -out_ptr[i],
				       dim - i - 1);
		}
	}
	else
	{
		invdiag = 1.0 / diag;
		for (; i < dim; i++)
		{
			out_ptr[i] *= invdiag;
			mxDoubleMulAdd__(&out_ptr[i + 1], &mat_ptr[i][i + 1], -out_ptr[i],
				       dim - i - 1);
		}
	}
}

/* Dsolve -- solves Dx=b where D is the diagonal of A -- may be in-situ */
void mxDSolve(Mat * mat, Vec * b, Vec * x)
{
	int dim, i;
	double tiny;

	dim = min(mat->m, mat->n);

	tiny = 10.0 / INF;

	dim = b->n;
	for (i = 0; i < dim; i++)
	{
		if (fabs(mat->ptr[i][i]) <= tiny * fabs(b->ptr[i]))
			error("DSolve: sorry, singular problem.");
		else
			x->ptr[i] = b->ptr[i] / mat->ptr[i][i];
	}
}

/* LTsolve -- back substitution with optional over-riding diagonal
		using the LOWER triangular part of matrix
		-- can be in-situ but doesn't need to be */
void mxLTSolve(Mat * mat, Vec * b, Vec * out, double diag)
{
	int dim;
	int i, i_lim;
	double **mat_ptr, *b_ptr, *out_ptr, tmp, invdiag, tiny;

	dim = min(mat->m, mat->n);
	mat_ptr = mat->ptr;
	b_ptr = b->ptr;
	out_ptr = out->ptr;

	tiny = 10.0 / INF;

	for (i = dim - 1; i >= 0; i--)
	{
		if (b_ptr[i] != 0.0)
			break;
	}
	i_lim = i;

	if (b != out)
	{
		mxZero__(out_ptr, out->n);
		double *ptr_r = b_ptr;
		double *ptr_w = out_ptr;
		for (int j = 0; j < i_lim + 1; j++)
			*ptr_w++ = *ptr_r++;
	}

	if (diag == 0.0)
	{
		for (; i >= 0; i--)
		{
			tmp = mat_ptr[i][i];
			if (fabs(tmp) <= tiny * fabs(out_ptr[i]))
				error("LTSolve: sorry, singular problem.");
			out_ptr[i] /= tmp;
			mxDoubleMulAdd__(out_ptr, mat_ptr[i], -out_ptr[i], i);
		}
	}
	else
	{
		invdiag = 1.0 / diag;
		for (; i >= 0; i--)
		{
			out_ptr[i] *= invdiag;
			mxDoubleMulAdd__(out_ptr, mat_ptr[i], -out_ptr[i], i);
		}
	}
}

}

