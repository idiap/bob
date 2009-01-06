#include "mx_givens.h"
#include <math.h>

namespace Torch {

/*
		Files for matrix computations

	Givens operations file. Contains routines for calculating and
	applying givens rotations for/to vectors and also to matrices by
	row and by column.
*/

/* givens -- returns c,s parameters for Givens rotation to
		eliminate y in the vector [ x y ]' */
void mx_givens(double x, double y, double * c, double * s)
{
	double norm = sqrt(x * x + y * y);
	if (norm == 0.0)
	{
		*c = 1.0;
		*s = 0.0;
	}				/* identity */
	else
	{
		*c = x / norm;
		*s = y / norm;
	}
}

/* rot_vec -- apply Givens rotation to x's i & k components */
void mx_rot_vec(Vec * x, int i, int k, double c, double s, Vec * out)
{
	out->copy(x);

	double temp = c * out->ptr[i] + s * out->ptr[k];
	out->ptr[k] = -s * out->ptr[i] + c * out->ptr[k];
	out->ptr[i] = temp;
}

/* rot_rows -- premultiply mat by givens rotation described by c,s */
void mx_rot_rows(Mat * mat, int i, int k, double c, double s, Mat * out)
{
	out->copy(mat);

	for (int j = 0; j < mat->n; j++)
	{
		double temp = c * out->ptr[i][j] + s * out->ptr[k][j];
		out->ptr[k][j] = -s * out->ptr[i][j] + c * out->ptr[k][j];
		out->ptr[i][j] = temp;
	}
}

/* rot_cols -- postmultiply mat by givens rotation described by c,s */
void mx_rot_cols(Mat * mat, int i, int k, double c, double s, Mat * out)
{
	out->copy(mat);

	for (int j = 0; j < mat->m; j++)
	{
		double temp = c * out->ptr[j][i] + s * out->ptr[j][k];
		out->ptr[j][k] = -s * out->ptr[j][i] + c * out->ptr[j][k];
		out->ptr[j][i] = temp;
	}
}

}

