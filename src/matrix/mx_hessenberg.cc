#include "mx_hessenberg.h"
#include "mx_householder.h"

namespace Torch {

/*
		File containing routines for determining Hessenberg
	factorisations.
*/

/* Hfactor -- compute Hessenberg factorisation in compact form.
	-- factorisation performed in situ
	-- for details of the compact form see QRfactor.c and matrix2.doc */
void mxHFactor(Mat * mat, Vec * diag, Vec * beta)
{
  int limit = mat->m - 1;

  Vec *tmp = new Vec(mat->m);

  for (int k = 0; k < limit; k++)
  {
    mat->getCol(k, tmp);
    mxHhVec(tmp, k + 1, &beta->ptr[k], tmp, &mat->ptr[k + 1][k]);
    diag->ptr[k] = tmp->ptr[k + 1];
    mxHhTrCols(mat, k + 1, k + 1, tmp, beta->ptr[k]);
    mxHhTrRows(mat, 0, k + 1, tmp, beta->ptr[k]);
  }

  delete tmp;
}

/* makeHQ -- construct the Hessenberg orthogonalising matrix Q;
	-- i.e. Hess M = Q.M.Q'	*/
void mxMakeHQ(Mat * h_mat, Vec * diag, Vec * beta, Mat * q_out)
{
  int limit = h_mat->m - 1;
//    Qout = m_resize(Qout,H->m,H->m);

  Vec *tmp1 = new Vec(h_mat->m);
  Vec *tmp2 = new Vec(h_mat->m);

  for (int i = 0; i < h_mat->m; i++)
  {
    tmp1->zero();
    tmp1->ptr[i] = 1.0;

    /* apply H/h transforms in reverse order */
    for (int j = limit - 1; j >= 0; j--)
    {
      h_mat->getCol(j, tmp2);
      tmp2->ptr[j + 1] = diag->ptr[j];
      mxHhTrVec(tmp2, beta->ptr[j], j + 1, tmp1, tmp1);
    }

    /* insert into Qout */
    q_out->setCol(i, tmp1);
  }
  delete tmp1;
  delete tmp2;
}

/* makeH -- construct actual Hessenberg matrix */
void mxMakeH(Mat * h_mat, Mat * h_out)
{
//    Hout = m_resize(Hout,H->m,H->m);
  h_out->copy(h_mat);

  int limit = h_mat->m;
  for (int i = 1; i < limit; i++)
  {
    for (int j = 0; j < i - 1; j++)
      h_out->ptr[i][j] = 0.0;
  }
}

}

