/**
 * @author <a href="mailto:andre.dos.anjos@gmail.com">Andre Anjos</a> 
 * @date Tue  8 Mar 08:20:02 2011 
 *
 * @brief Defines the HornAndSchunckFlow methods
 */

#include "ip/HornAndSchunckFlow.h"

namespace of = Torch::ip::optflow;

/**
 * Compares two arrays for their shape
 */
template<typename T1, typename T2>
static bool shapeq(const blitz::Array<T1,2>& a1, const blitz::Array<T2,2>& a2) {
  return (a1.rows() == a2.rows()) && (a1.columns() == a2.columns());
}

template<typename T1, typename T2>
static bool shapeq_plus(const blitz::Array<T1,2>& a1,
    const blitz::Array<T2,2>& a2, int plus) {
  return (a1.rows()+plus == a2.rows()) && (a1.columns()+plus == a2.columns());
}

/**
 * Estimates the partial derivative in the x direction. The formula is:
 *
 * Ex = 1/4 { E(i,j+1,k) - E(i,j,k) + 
 *            E(i+1,j+1,k) - E(i+1,j,k) +
 *            E(i,j+1,k+1) - E(i,j,k+1) +
 *            E(i+1,j+1,k+1) - E(i+1,j,k+1) }
 *
 * Please note we have separate images for t = k (i1) and t = k+1 (i2). So we
 * can re-write like this:
 * 
 * Ex = 1/4 { i1(i,j+1) - i1(i,j) +
 *            i1(i+1,j+1) - i1(i+1,j) +
 *            i2(i,j+1) - i2(i,j) +
 *            i2(i+1,j+1) - i2(i+1,j) }
 *
 * Please note that this formula will not be bound for the last column and the
 * last row of the image and what to do is lacking at the paper. Current
 * implementations replicate the last row and column of the input image. This
 * effectively sets the Ex flow on the last column to be zero.
 */
static void Ex(const blitz::Array<uint8_t,2>& i1, 
    const blitz::Array<uint8_t,2>& i2, blitz::Array<double,2>& result) {
  blitz::Range i(0,i1.extent(0)-2), j(0,i1.extent(1)-2); //avoids last col/row
  result(i,j) = 0.25 * ( i1(i,j+1) - i1(i,j) +
                         i1(i+1,j+1) - i1(i+1,j) +
                         i2(i,j+1) - i2(i,j) +
                         i2(i+1,j+1) - i2(i+1,j) );

  //now the calculations for the last row: because we wish to replicate the
  //last row to make the calculation, this is the same as considering twice the
  //terms in 'i' (since terms in 'i+1' will be the same)
  const int l = i1.extent(0)-1;
  result(l,j) = 0.5 * (i1(l,j+1) - i1(l,j) + i2(l,j+1) - i2(l,j));

  //last column is zero
  result(blitz::Range::all(),i1.extent(1)-1) = 0;
}

/**
 * Estimates the partial derivative in the y direction. The formula is:
 *
 * Ey = 1/4 { E(i+1,j,k) - E(i,j,k) + 
 *            E(i+1,j+1,k) - E(i,j+1,k) +
 *            E(i+1,j,k+1) - E(i,j,k+1) +
 *            E(i+1,j+1,k+1) - E(i,j+1,k+1) }
 *
 * Please note we have separate images for t = k (i1) and t = k+1 (i2). So we
 * can re-write like this:
 * 
 * Ey = 1/4 { i1(i+1,j) - i1(i,j) +
 *            i1(i+1,j+1) - i1(i,j+1) +
 *            i2(i+1,j) - i2(i,j) }
 *            i2(i+1,j+1) - i2(i,j+1) }
 *
 * Please note that this formula will not be bound for the last column and the
 * last row of the image and what to do is lacking at the paper. Current
 * implementations replicate the last row and column of the input image. This
 * effectively sets the Ey flow on the last row to be zero.
 */
static void Ey(const blitz::Array<uint8_t,2>& i1, 
    const blitz::Array<uint8_t,2>& i2, blitz::Array<double,2>& result) {
  blitz::Range i(0,i1.extent(0)-2), j(0,i1.extent(1)-2); //avoids last col/row
  result(i,j) = 0.25 * ( i1(i+1,j) - i1(i,j) +
                         i1(i+1,j+1) - i1(i,j+1) +
                         i2(i+1,j) - i2(i,j) +
                         i2(i+1,j+1) - i2(i,j+1) );

  //now the calculations for the last column: because we wish to replicate the
  //last col to make the calculation, this is the same as considering twice the
  //terms in 'j' (since terms in 'j+1' will be the same)
  const int l = i1.extent(1)-1;
  result(i,l) = 0.5 * (i1(i+1,l) - i1(i,l) + i2(i+1,l) - i2(i,l));

  //last row is zero
  result(i1.extent(0)-1,blitz::Range::all()) = 0;
}

/**
 * Estimates the partial derivative in the t (time) direction. The formula is:
 *
 * Et = 1/4 { E(i,j,k+1) - E(i,j,k) + 
 *            E(i+1,j,k+1) - E(i+1,j,k) +
 *            E(i,j+1,k+1) - E(i,j+1,k) +
 *            E(i+1,j+1,k+1) - E(i+1,j+1,k) }
 *
 * Please note we have separate images for t = k (i1) and t = k+1 (i2). So we
 * can re-write like this:
 * 
 * Et = 1/4 { i2(i,j) - i1(i,j) +
 *            i2(i+1,j) - i1(i+1,j) +
 *            i2(i,j+1) - i1(i,j+1) }
 *            i2(i+1,j+1) - i1(i+1,j+1) }
 *
 * Please note that this formula will not be bound for the last column and the
 * last row of the image and what to do is lacking at the paper. Current
 * implementations replicate the last row and column of the input image. 
 */
static void Et(const blitz::Array<uint8_t,2>& i1, 
    const blitz::Array<uint8_t,2>& i2, blitz::Array<double,2>& result) {
  blitz::Range i(0,i1.extent(0)-2), j(0,i1.extent(1)-2); //avoids last col/row
  result(i,j) = 0.25 * ( i2(i,j) - i1(i,j) +
                         i2(i+1,j) - i1(i+1,j) +
                         i2(i,j+1) - i1(i,j+1) +
                         i2(i+1,j+1) - i1(i+1,j+1) );

  //now the calculations for the last row: because we wish to replicate the
  //last row to make the calculation, this is the same as considering twice the
  //terms in 'i' (since terms in 'i+1' will be the same)
  const int l = i1.extent(0)-1;
  result(l,j) = 0.5 * (i2(l,j)-i1(l,j) + i2(l,j+1)-i1(l,j+1));

  //a similar analysis for the last colum
  const int k = i1.extent(1)-1;
  result(i,k) = 0.5 * (i2(i,k)-i1(i,k) + i2(i+1,k)-i1(i+1,k));

  //the very last pixel in the last row/column is zero because of all the
  //previous considerations
  result(l,k) = 0;
}

/**
 * Estimates the laplacian u/v bar components in the iterative formula. The
 * estimation can be depicted as a convolution between the current estimate "u"
 * and a kernel that approximates the Laplacian. The paper suggests the
 * following kernel:
 *
 * [1/12 1/6 1/12]
 * [1/6  -1  1/6 ]
 * [1/12 1/6 1/12]
 *
 * Which leads us to the following equation in "u" (excluding the middle
 * component):
 *
 * Ubar(n) = 1/6  { u(i-1,j,k) + u(i,j+1,k) + u(i+1,j,k) + u(i,j-1,k) } + 
 *           1/12 { u(i-1,j-1,k) + u(i-1,j+1,k) + u(i+1,j+1,k) + u(i+1,j-1,k) }
 *
 * Common literature on the web and implementations in matlab for the Horn &
 * Schunck method use a different approximation though:
 *
 * [0  1  0]    [-1 -1 -1]
 * [1 -4  1] or [-1  8 -1]
 * [0  1  0]    [-1 -1 -1]
 *
 * OpenCV uses the first and normalizes the results by 1/4. That is faster and
 * we will do (approximately) the same.
 *
 * This quantity is only calculated for the current image being analyzed. In
 * the case of our nomenclature, we are talking about i1 (t = k).
 *
 * Please note that the above formulas only exists in a subrange of the input
 * velocities (namely 1:extent-1), because it has a 1 pixel window. In the
 * first/last column/rows, we replicate the column/row to make the estimate.
 */
static void laplacian_border(const blitz::Array<double,2>& u, 
    blitz::Array<double,2>& result) {
  blitz::Range i(1,u.extent(0)-2), j(1,u.extent(1)-2); //avoids first/last r/c
  result(i,j) = 0.25 * ( u(i-1,j) + u(i,j+1) + u(i+1,j) + u(i,j-1) ); 
  
  const int lr = u.extent(0)-1; //last row index
  const int lc = u.extent(1)-1; //last column index

  //middle of first row: i-1 is not bound
  result(0,j)  = 0.25 * ( u(0,j) + u(0,j+1) + u(1,j) + u(0,j-1) ); 
  //middle of last row: i+1 is not bound
  result(lr,j) = 0.25 * ( u(lr-1,j) + u(lr,j+1) + u(lr,j) + u(lr,j-1) ); 
  //middle of first column: j-1 is not bound
  result(i,0)  = 0.25 * ( u(i-1,0) + u(i,1) + u(i+1,0) + u(i,0) );
  //middle of last column: j+1 is not bound
  result(i,lc) = 0.25 * ( u(i-1,lc) + u(i,lc) + u(i+1,lc) + u(i,lc-1) );

  //corner pixels
  result(0,0)   = 0.25 * (2*u(0,0) + u(0,1) + u(1,0)); //top-left
  result(0,lc)  = 0.25 * (2*u(0,lc) + u(0,lc-1) + u(1,lc)); //top-right
  result(lr,0)  = 0.25 * (2*u(lr,0) + u(lr-1,0) + u(lr,1)); //bottom-left
  result(lr,lc) = 0.25 * (2*u(lr,lc) + u(lr-1,lc) + u(lr,lc-1)); //bottom-right
}

/**
 * This is the original implementation for Horn & Schunck, in case you want to
 * try it out.
 */
/*
static void laplacian_border(const blitz::Array<double,2>& u, 
    blitz::Array<double,2>& result) {
  blitz::Range i(1,u.extent(0)-2), j(1,u.extent(1)-2);
  result = 0;
  result = (1.0/6) *  ( u(i-1,j) + u(i,j+1) + u(i+1,j) + u(i,j-1) ) +
           (1.0/12) * ( u(i-1,j-1) + u(i-1,j+1) + u(i+1,j+1), u(i+1,j-1) );
}
*/

void of::evalHornAndSchunckFlow(double alpha, size_t iterations,
 const blitz::Array<uint8_t,2>& i1, const blitz::Array<uint8_t,2>& i2,
 blitz::Array<double,2>& u0, blitz::Array<double,2>& v0) {

  //we need some caching variables

  //caches for partial derivatives and laplacian estimators (averages)
  blitz::Array<double,2> ex(i1.shape());
  blitz::Array<double,2> ey(i1.shape());
  blitz::Array<double,2> et(i1.shape());
  blitz::Array<double,2> u(i1.shape());
  blitz::Array<double,2> v(i1.shape());
  
  //finally, a cache for the common term in the iterative formula
  blitz::Array<double, 2> cterm(i1.shape());

  //iterative flow calculation proposed by Horn & Schunck
  Ex(i1, i2, ex);
  Ey(i1, i2, ey);
  Et(i1, i2, et);
  double a2 = std::pow(alpha, 2);
  for (size_t i=0; i<iterations; ++i) {
    laplacian_border(u0, u);
    laplacian_border(v0, v);
    cterm = (ex*u + ey*v + et) / (blitz::pow2(ex) + blitz::pow2(ey) + a2);
    u0 = u - ex*cterm;
    v0 = v - ey*cterm;
  }
}

void of::evalHornAndSchunckEc2
(const blitz::Array<double,2>& u, const blitz::Array<double,2>& v,
 blitz::Array<double,2>& error) {
  blitz::Array<double,2> ux(u.shape());
  blitz::Array<double,2> vx(u.shape());
  laplacian_border(u, ux);
  laplacian_border(v, vx);
  if (!shapeq(u, error)) error.resize(u.shape());
  error = blitz::pow2(ux - u) + blitz::pow2(vx - v);
}

void of::evalHornAndSchunckEb
(const blitz::Array<uint8_t,2>& i1, const blitz::Array<uint8_t,2>& i2,
 const blitz::Array<double,2>& u, const blitz::Array<double,2>& v,
 blitz::Array<double,2>& error) {
  blitz::Array<double,2> ex(i1.shape());
  blitz::Array<double,2> ey(i1.shape());
  blitz::Array<double,2> et(i1.shape());
  Ex(i1, i2, ex);
  Ey(i1, i2, ey);
  Et(i1, i2, et);
  if (!shapeq(u, error)) error.resize(u.shape());
  error = ex*u + ey*v + et;
}
