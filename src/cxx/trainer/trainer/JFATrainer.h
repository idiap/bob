/**
 * @author Laurent El-Shafey <Laurent.El-Shafey@idiap.ch>
 * @date Thu 14 Jul 2011 16:08:30
 *
 * @brief JFA functions
 */

#ifndef TORCH5SPRO_TRAINER_JFATRAINER_H
#define TORCH5SPRO_TRAINER_JFATRAINER_H

#include <blitz/array.h>
#include "io/Arrayset.h"

namespace Torch { namespace trainer { namespace jfa {

/**
  C++ port of the JFA Cookbook
*/

/**
  @brief Updates eigenchannels (or eigenvoirces) from accumulators A and C. 

  @param A  An array of 2D square matrices (c x ru x ru)
  @param C  A 2D matrix (ru x cd)
  @param uv A 2D matrix (ru x cd)

  @warning  ru: rank of the matrix of eigenchannels u
            cd: size of the supervector
*/
void updateEigen(const blitz::Array<double,3> &A, const blitz::Array<double,2> &C,
  blitz::Array<double,2> &uv);

/**
  @brief Provides new estimates of channel factors x given zeroth and first 
    order sufficient statistics (N. F), current hyper-parameters of joint 
    factor analysis  model (m, E, d, u, v) and current estimates of speaker 
    and channel factors (x, y, z), for a joint factor analysis model.

  @param F 2D matrix (T x CD) of first order statistics (not centered). 
    The rows correspond to training segments (T segments). 
    The number of columns is given by the supervector dimensionality. 
    The first n columns correspond to the n dimensions of the first Gaussian 
    component, the second n columns to second component, and so on.

  @param N 2D matrix (TxC) of zero order statistics (occupation counts of 
    Gaussian components). 
    The rows correspond to training segments. 
    The columns correspond to Gaussian components.

  @param m 1D speaker and channel independent mean supervector (CD)
    (e.g. concatenated UBM mean vectors)

  @param E 1D speaker and channel independent variance supervector (CD)
    (e.g. concatenated UBM variance vectors)

  @param d 1D row vector (CD) that is the diagonal from the diagonal matrix 
    describing the remaining speaker variability (not described by 
    eigenvoices). 
    The number of columns is given by the supervector dimensionality.

  @param v 2D matrix of eigenvoices (rv x CD)
    The rows of matrix v are 'eigenvoices'. (The number of rows must be the
    same as the number of columns of matrix y). 
    The number of columns is given by the supervector dimensionality.

  @param u 2D matrix of eigenchannels (ru x CD)
    The rows of matrix u are 'eigenchannels'. (The number of rows must be the
    same as the number of columns of matrix x).
    Number of columns is given by the supervector dimensionality.

   @param z 2D matrix of speaker factors corresponding to matrix d (Nspeaker x CD)
    The rows correspond to speakers (values in vector spk_ids are the indices 
    of the rows, therfore the number of the rows must be (at least) the highest
    value in spk_ids). Number of columns is given by the supervector 
    dimensionality.
 
  @param y 2D matrix of speaker factors corresponding to eigenvoices (Nspeaker x rv)
    The rows correspond to speakers (values in vector spk_ids are the indices 
    of the rows, therfore the number of the rows must be (at least) the 
    highest value in spk_ids). 
    The columns correspond to eigenvoices (The number of columns must the same 
    as the number of rows of matrix v).

  @param x 2D matrix of speaker factors for eigenchannels (Nspeaker x ru)
    The rows correspond to training segments. 
    The columns correspond to eigenchannels (The number of columns must be 
    the same as the number of rows of matrix u)

  @param spk_ids 1D column vector (T) with rows corresponding to training 
    segments and integer values identifying a speaker. Rows having same values
    identifies segments spoken by same speakers. 
    The values are indices of rows in y and z matrices containing 
    corresponding speaker factors.
  @warning Rows corresponding to the same speaker SHOULD be consecutive.
*/
void estimateXandU(const blitz::Array<double,2> &F, const blitz::Array<double,2> &N,
  const blitz::Array<double,1> &m, const blitz::Array<double,1> &E, 
  const blitz::Array<double,1> &d, const blitz::Array<double,2> &v, 
  const blitz::Array<double,2> &u, const blitz::Array<double,2> &z, 
  const blitz::Array<double,2> &y, blitz::Array<double,2> &x,
  const blitz::Array<uint32_t,1> &spk_ids);



void estimateYandV(const blitz::Array<double,2> &F, const blitz::Array<double,2> &N,
  const blitz::Array<double,1> &m, const blitz::Array<double,1> &E, 
  const blitz::Array<double,1> &d, const blitz::Array<double,2> &v, 
  const blitz::Array<double,2> &u, const blitz::Array<double,2> &z, 
  blitz::Array<double,2> &y, const blitz::Array<double,2> &x,
  const blitz::Array<uint32_t,1> &spk_ids);

void estimateZandD(const blitz::Array<double,2> &F, const blitz::Array<double,2> &N,
  const blitz::Array<double,1> &m, const blitz::Array<double,1> &E, 
  const blitz::Array<double,1> &d, const blitz::Array<double,2> &v, 
  const blitz::Array<double,2> &u, blitz::Array<double,2> &z, 
  const blitz::Array<double,2> &y, const blitz::Array<double,2> &x,
  const blitz::Array<uint32_t,1> &spk_ids);

}}}

#endif /* TORCH5SPRO_TRAINER_JFATRAINER_H */
