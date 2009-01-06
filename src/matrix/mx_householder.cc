#include "mx_householder.h"
#include "mx_low_level.h"

namespace Torch {

/*
  Files for matrix computations

  Householder transformation file. Contains routines for calculating
  householder transformations, applying them to vectors and matrices
  by both row & column.
*/


/* hhvec -- calulates Householder vector to eliminate all entries after the
	i0 entry of the vector vec. It is returned as out. May be in-situ */
void mxHhVec(Vec * vec, int i0, double * beta, Vec * out, double * newval)
{
	out->copy(out, i0);
	double norm = sqrt(out->iP(out, i0));
	if (norm <= 0.0)
	{
		*beta = 0.0;
		return;
	}
	*beta = 1.0 / (norm * (norm + fabs(out->ptr[i0])));
	if (out->ptr[i0] > 0.0)
		*newval = -norm;
	else
		*newval = norm;
	out->ptr[i0] -= *newval;
}

/* hhtrvec -- apply Householder transformation to vector -- may be in-situ */
/* hh = Householder vector */
void mxHhTrVec(Vec * hh, double beta, int i0, Vec * in, Vec * out)
{
	double scale = beta * hh->iP(in, i0);
	out->copy(in);
	mxDoubleMulAdd__(&out->ptr[i0], &hh->ptr[i0], -scale, in->n - i0);
}

/* hhtrrows -- transform a matrix by a Householder vector by rows
	starting at row i0 from column j0 -- in-situ */
void mxHhTrRows(Mat * mat, int i0, int j0, Vec * hh, double beta)
{
	double ip, scale;

	if (beta == 0.0)
		return;

	/* for each row ... */
	for (int i = i0; i < mat->m; i++)
	{				/* compute inner product */
		ip = mxIp__(&mat->ptr[i][j0], &hh->ptr[j0], mat->n - j0);

		scale = beta * ip;
		if (scale == 0.0)
			continue;

		/* do operation */
		mxDoubleMulAdd__(&mat->ptr[i][j0], &hh->ptr[j0], -scale, mat->n - j0);
	}
}


/* hhtrcols -- transform a matrix by a Householder vector by columns
	starting at row i0 from column j0 -- in-situ */
void mxHhTrCols(Mat * mat, int i0, int j0, Vec * hh, double beta)
{
	if (beta == 0.0)
		return;

	Vec *w = new Vec(mat->n);
	w->zero();

	for (int i = i0; i < mat->m; i++)
	{
		if (hh->ptr[i] != 0.0)
			mxDoubleMulAdd__(&w->ptr[j0], &mat->ptr[i][j0], hh->ptr[i],
					 mat->n - j0);
	}
	for (int i = i0; i < mat->m; i++)
	{
		if (hh->ptr[i] != 0.0)
			mxDoubleMulAdd__(&mat->ptr[i][j0], &w->ptr[j0], -beta * hh->ptr[i],
					 mat->n - j0);
	}

	delete w;
}

}

