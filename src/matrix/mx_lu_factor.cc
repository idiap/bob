#include "Perm_operations.h"
#include "mx_lu_factor.h"
#include "mx_low_level.h"
#include "mx_solve.h"
#include <math.h>

#ifndef INF
	#define INF	(10000000000.0)
#endif

namespace Torch {

/*
 * Most matrix factorisation routines are in-situ unless otherwise specified
 */

/*
 * LUfactor -- gaussian elimination with scaled partial pivoting -- Note:
 * returns LU matrix which is A
 */
void mxLUFactor(Mat * mat, Perm * pivot)
{
	int m, n;
	int i_max;
	double **mat_v, *mat_piv, *mat_row;
	double max1, temp, tiny;

	Vec *scale = new Vec(mat->m);

	m = mat->m;
	n = mat->n;
	mat_v = mat->ptr;

	tiny = 10.0 / INF;

	/*
	 * initialise pivot with identity permutation
	 */
	for (int i = 0; i < m; i++)
		pivot->ptr[i] = i;

	/*
	 * set scale parameters
	 */
	for (int i = 0; i < m; i++)
	{
		max1 = 0.0;
		for (int j = 0; j < n; j++)
		{
			temp = fabs(mat_v[i][j]);
			if (max1 < temp)
				max1 = temp;
		}
		scale->ptr[i] = max1;
	}

	/*
	 * main loop
	 */
	int k_max = (m < n ? m : n) - 1;
	for (int k = 0; k < k_max; k++)
	{
		/*
		 * find best pivot row
		 */
		max1 = 0.0;
		i_max = -1;
		for (int i = k; i < m; i++)
		{
			if (fabs(scale->ptr[i]) >= tiny * fabs(mat_v[i][k]))
			{
				temp = fabs(mat_v[i][k]) / scale->ptr[i];
				if (temp > max1)
				{
					max1 = temp;
					i_max = i;
				}
			}
		}

		/*
		 * if no pivot then ignore column k...
		 */
		if (i_max == -1)
		{
			/*
			 * set pivot entry mat[k][k] exactly to zero, rather than just
			 * "small"
			 */
			mat_v[k][k] = 0.0;
			continue;
		}

		/*
		 * do we pivot ?
		 */
		if (i_max != k)
		{				/*
				   * yes we do...
				 */
			mxTrPerm(pivot, i_max, k);
			for (int j = 0; j < n; j++)
			{
				temp = mat_v[i_max][j];
				mat_v[i_max][j] = mat_v[k][j];
				mat_v[k][j] = temp;
			}
		}

		/*
		 * row operations
		 */
		for (int i = k + 1; i < m; i++)
		{				/*
				   * for each row do...
 *//*
 * Note: divide by zero should never happen
 */
			temp = mat_v[i][k] = mat_v[i][k] / mat_v[k][k];
			mat_piv = &(mat_v[k][k + 1]);
			mat_row = &(mat_v[i][k + 1]);
			if (k + 1 < n)
				mxDoubleMulAdd__(mat_row, mat_piv, -temp, n - (k + 1));
		}
	}

	delete scale;
}


/*
 * LUsolve -- given an LU factorisation in A, solve Ax=b
 */
void mxLUSolve(Mat * mat, Perm * pivot, Vec * b, Vec * x)
{
	// x := P.b
	mxPermVec(pivot, b, x);
	// implicit diagonal = 1
	mxLSolve(mat, x, x, 1.0);
	// explicit diagonal
	mxUSolve(mat, x, x, 0.0);
}

/*
 * LUTsolve -- given an LU factorisation in A, solve A^T.x=b
 */
void mxLUTSolve(Mat * mat, Perm * pivot, Vec * b, Vec * x)
{
	x->copy(b);
	// explicit diagonal
	mxUTSolve(mat, x, x, 0.0);
	// implicit diagonal = 1
	mxLTSolve(mat, x, x, 1.0);
	// x := P^T.tmp
	mxPermInvVec(pivot, x, x);
}

/*
 * m_inverse -- returns inverse of A, provided A is not too rank deficient --
 * uses LU factorisation
 */
void mxInverse(Mat * mat, Mat * out)
{
	// That's me...
	Mat *mat_cp = new Mat(mat->m, mat->n);
	Vec *tmp = new Vec(mat->m);
	Vec *tmp2 = new Vec(mat->m);
	Perm *pivot = new Perm(mat->m);

	mat_cp->copy(mat);
	mxLUFactor(mat_cp, pivot);

	for (int i = 0; i < mat->n; i++)
	{
		tmp->zero();
		tmp->ptr[i] = 1.0;
		mxLUSolve(mat_cp, pivot, tmp, tmp2);
		out->setCol(i, tmp2);
	}

	delete mat_cp;
	delete tmp;
	delete tmp2;
	delete pivot;
}

/*
 * LUcondest -- returns an estimate of the condition number of LU given the
 * LU factorisation in compact form
 */
double mxLUCondest(Mat * mat, Perm * pivot)
{
	double cond_est, L_norm, U_norm, sum, tiny;
	int n = mat->n;
	Vec *y = new Vec(n);
	Vec *z = new Vec(n);

	tiny = 10.0 / INF;

	for (int i = 0; i < n; i++)
	{
		sum = 0.0;
		for (int j = 0; j < i; j++)
			sum -= mat->ptr[j][i] * y->ptr[j];
		sum -= (sum < 0.0) ? 1.0 : -1.0;
		if (fabs(mat->ptr[i][i]) <= tiny * fabs(sum))
		{
			delete y;
			delete z;
			return INF;
		}
		y->ptr[i] = sum / mat->ptr[i][i];
	}

	mxLTSolve(mat, y, y, 1.0);
	mxLUSolve(mat, pivot, y, z);

	/*
	 * now estimate norm of A (even though it is not directly available)
	 */
	/*
	 * actually computes ||L||_inf.||U||_inf
	 */
	U_norm = 0.0;
	for (int i = 0; i < n; i++)
	{
		sum = 0.0;
		for (int j = i; j < n; j++)
			sum += fabs(mat->ptr[i][j]);
		if (sum > U_norm)
			U_norm = sum;
	}
	L_norm = 0.0;
	for (int i = 0; i < n; i++)
	{
		sum = 1.0;
		for (int j = 0; j < i; j++)
			sum += fabs(mat->ptr[i][j]);
		if (sum > L_norm)
			L_norm = sum;
	}

	cond_est = U_norm * L_norm * z->normInf() / y->normInf();

	delete y;
	delete z;

	return cond_est;
}

/*
   Given #A# and #b#, solve #A.x=b# */
void mxSolve(Mat *mat, Vec *b, Vec *x)
{
	Perm *pivot = new Perm(mat->m);
	mxLUFactor(mat, pivot);
	mxLUSolve(mat, pivot, b, x);
	delete pivot;
}

}

