/**
 * @file ip/cxx/test/SIFT.cc
 * @date Tue Sep 18 18:56:25 2012 +0200
 * @author Laurent El Shafey <Laurent.El-Shafey@idiap.ch>
 *
 * @brief Test the SIFT implementation for 2D arrays and compare against
 *   VLfeat one.
 *
 * Copyright (C) 2011-2012 Idiap Research Institute, Martigny, Switzerland
 * 
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, version 3 of the License.
 * 
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 * 
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE IP-SIFT Tests
#define BOOST_TEST_MAIN
#include <boost/test/unit_test.hpp>
#include <blitz/array.h>
#include <bob/ip/SIFT.h>

#include <algorithm>
#include <random/uniform.h>

struct T {
  double eps;
  T(): eps(1e-6) {}

  ~T() {}
};

template<typename T, typename U, int d>  
void check_dimensions(const blitz::Array<T,d>& t1, const blitz::Array<U,d>& t2)
{
  BOOST_REQUIRE_EQUAL(t1.dimensions(), t2.dimensions());
  for (int i=0; i<t1.dimensions(); ++i)
    BOOST_CHECK_EQUAL(t1.extent(i), t2.extent(i));
}

template<typename T, typename U>  
void checkBlitzClose(const blitz::Array<T,3>& t1, const blitz::Array<U,3>& t2,
  const double eps )
{
  check_dimensions(t1, t2);
  for (int i=0; i<t1.extent(0); ++i)
    for (int j=0; j<t1.extent(1); ++j)
      for (int k=0; k<t1.extent(2); ++k)
        BOOST_CHECK_SMALL( fabs( t1(i,j,k) - t2(i,j,k) ), eps );
}


// 1. Constants used for the comparison with VLfeat implementation
static const double MAGNIF=3.;
static const int NOCTAVES=1;
static const int NINTERVALS=3;
static const int OCTAVE_MIN=0; // 
static const int OCTAVE_CUR=0; // 
static const int HEIGHT=200;
static const int WIDTH=200;
static const int OCTAVE_HEIGHT=HEIGHT/(1<<OCTAVE_CUR);
static const int OCTAVE_WIDTH=200/(1<<OCTAVE_CUR);
static const int NBP=4;
static const int NBO=8;
static const float VL_EPSILON_F=1.19209290E-07F;
static const double VL_EPSILON_D=2.220446049250313e-16;
static const double WINDOW_SIZE=NBP / 2 ;
static const double SIGMA0=1.6;
static bob::ip::GSSKeypoint kp(2., 50, 70);

// 2. Friend class for testing bob SIFT implementation (final part of descriptor computation)
namespace bob { namespace ip {
  class SIFTtest
  {
    public:
      void run(const blitz::Array<double,2>& src, blitz::Array<double,3>& descr);
  };
}}

void bob::ip::SIFTtest::run(const blitz::Array<double,2>& src, blitz::Array<double,3>& descr)
{
  // 1. Initializes a SIFT object
  bob::ip::SIFT op(HEIGHT, WIDTH, NOCTAVES, NINTERVALS, OCTAVE_MIN);

  // 2. Initializes a keypoint information structure
  bob::ip::GSSKeypointInfo kpi;
  int o=0;
  kpi.o=0;
  kpi.s=1;
  // (y,x) coordinates
  const double factor = pow(2.,o);
  kpi.iy = (int)floor(kp.y/factor + 0.5);
  kpi.ix = (int)floor(kp.x/factor + 0.5);

  // Set input data and compute descriptor
  std::vector<blitz::Array<double,3> > vec;
  blitz::Array<double,3> pyr(NINTERVALS+3,HEIGHT,WIDTH);
  blitz::Range rall = blitz::Range::all();
  for (int i=0; i<NINTERVALS+3; ++i)
    pyr(i,rall,rall) = src;
  vec.push_back(pyr);
  op.setGaussianPyramid(vec);
  op.computeGradient();
  op.computeDescriptor(kp, kpi, descr);
}

// 3. Code extracted from VLFeat for testing purposes and updated for our
// particular comparison. This extracted code was originally distributed 
// under a BSD license.
float vl_mod_2pi_f(float x)
{
  while(x > (float)(2 * M_PI)) x -= (float) (2 * M_PI);
  while(x < 0.0F) x += (float) (2 * M_PI);
  return x; 
}

float vl_abs_f(float x)
{
  return fabsf(x);
}

#define VL_MIN(x,y) (((x)<(y))?(x):(y))
#define VL_MAX(x,y) (((x)>(y))?(x):(y))

// The blitz::Array should have been C-style allocated!
void vl_update_gradient(const blitz::Array<double,2>& src_, blitz::Array<float,3>& grad_)
{
  int const s_min=-1;
  const int w = OCTAVE_WIDTH;
  const int h = OCTAVE_HEIGHT;
  int const xo    = 1 ;
  int const yo    = w ;
  int const so    = h * w ;
  int y, s ;

  const double * src_c = src_.data();
  float * grad_c = grad_.data();

  // SINGLE scale s=0
  s = 0;
  {

    const double * src, *end;
    float *grad, gx, gy ;

#define SAVE_BACK                                           \
    *grad++ = sqrt (gx*gx + gy*gy) ;                        \
    *grad++ = vl_mod_2pi_f   (atan2 (gy, gx) + 2*M_PI) ;    \
    ++src ;                                                 

    src = src_c;
    grad = grad_c + 2 * so * (s - s_min -1) ;

    // first pixel of the first row
    gx = src[+xo] - src[0] ;
    gy = src[+yo] - src[0] ;
    SAVE_BACK ;

    // middle pixels of the  first row
    end = (src - 1) + w - 1 ;
    while (src < end) {
      gx = 0.5 * (src[+xo] - src[-xo]) ;
      gy =        src[+yo] - src[0] ;
      SAVE_BACK ;
    }

    // last pixel of the first row
    gx = src[0]   - src[-xo] ;
    gy = src[+yo] - src[0] ;
    SAVE_BACK ;

    for (y = 1 ; y < h -1 ; ++y) {

      // first pixel of the middle rows
      gx =        src[+xo] - src[0] ;
      gy = 0.5 * (src[+yo] - src[-yo]) ;
      SAVE_BACK ;

      // middle pixels of the middle rows
      end = (src - 1) + w - 1 ;
      while (src < end) {
        gx = 0.5 * (src[+xo] - src[-xo]) ;
        gy = 0.5 * (src[+yo] - src[-yo]) ;
        SAVE_BACK ;
      }

      // last pixel of the middle row
      gx =        src[0]   - src[-xo] ;
      gy = 0.5 * (src[+yo] - src[-yo]) ;
      SAVE_BACK ;
    }

    // first pixel of the last row
    gx = src[+xo] - src[0] ;
    gy = src[  0] - src[-yo] ;
    SAVE_BACK ;

    // middle pixels of the last row
    end = (src - 1) + w - 1 ;
    while (src < end) {
      gx = 0.5 * (src[+xo] - src[-xo]) ;
      gy =        src[0]   - src[-yo] ;
      SAVE_BACK ;
    }

    // last pixel of the last row
    gx = src[0]   - src[-xo] ;
    gy = src[0]   - src[-yo] ;
    SAVE_BACK ;
  }
}
 
float vl_normalize_histogram(float *begin, float *end)
{
  float* iter ;
  float  norm = 0.0 ;

  for(iter = begin ; iter != end ; ++ iter)
    norm += (*iter) * (*iter) ;

  norm = sqrt(norm) + VL_EPSILON_F ;

  for(iter = begin; iter != end ; ++ iter)
    *iter /= norm ;

  return norm;
}

void vl_sift(const blitz::Array<double,2>& A,
  blitz::Array<float,3>& grad_, blitz::Array<float,3>& descr_)
{
  //   The SIFT descriptor is a three dimensional histogram of the
  //   position and orientation of the gradient.  There are NBP bins for
  //   each spatial dimension and NBO bins for the orientation dimension,
  //   for a total of NBP x NBP x NBO bins.
  //   The support of each spatial bin has an extension of SBP = 3sigma
  //   pixels, where sigma is the scale of the keypoint.  Thus all the
  //   bins together have a support SBP x NBP pixels wide. Since
  //   weighting and interpolation of pixel is used, the support extends
  //   by another half bin. Therefore, the support is a square window of
  //   SBP x (NBP + 1) pixels. Finally, since the patch can be
  //   arbitrarily rotated, we need to consider a window 2W += sqrt(2) x
  //   SBP x (NBP + 1) pixels wide.

  const float* grad = grad_.data();
  float* descr = descr_.data();

  double       xper        = pow(2.0, OCTAVE_CUR) ;

  int          w           = OCTAVE_WIDTH;
  int          h           = OCTAVE_HEIGHT;
  int const    xo          = 2 ;         // x-stride
  int const    yo          = 2 * w ;     // y-stride
  int const    so          = 2 * w * h ; // s-stride
  double       x           = kp.x     / xper ;
  double       y           = kp.y     / xper ;
  double       sigma       = kp.sigma / xper ;

  int          xi          = (int) (x + 0.5) ;
  int          yi          = (int) (y + 0.5) ;
  // Assumes scale of index 0
  int si = 0;

  // Move keypoint orientation into angle0 variable
  const double angle0 = kp.orientation;
  double const st0         = sin (angle0) ;
  double const ct0         = cos (angle0) ;
  double const SBP         = MAGNIF * sigma + VL_EPSILON_D ;
  int    const W           = floor
    (sqrt(2.0) * SBP * (NBP + 1) / 2.0 + 0.5) ;

  int const binto = 1 ;          // bin theta-stride
  int const binyo = NBO * NBP ;  // bin y-stride
  int const binxo = NBO ;        // bin x-stride

  int bin, dxi, dyi ;
  float const *pt ;
  float       *dpt ;

  // check bounds
  int const f_s_min=-1;
  int const f_s_max=NINTERVALS+1;
  if(
     xi    <  0               ||
     xi    >= w               ||
     yi    <  0               ||
     yi    >= h -    1        ||
     si    <  f_s_min + 1    ||
     si    >  f_s_max - 2     )
    return ;

  // synchronize gradient buffer
  vl_update_gradient(A,grad_);

  // clear descriptor
  memset(descr, 0, sizeof(float) * NBO*NBP*NBP) ;

  // Center the scale space and the descriptor on the current keypoint.
  // Note that dpt is pointing to the bin of center (SBP/2,SBP/2,0).
  pt  = grad + xi*xo + yi*yo + (si - f_s_min - 1)*so ;
  dpt = descr + (NBP/2) * binyo + (NBP/2) * binxo ;

#undef atd
#define atd(dbinx,dbiny,dbint) *(dpt + (dbint)*binto + (dbiny)*binyo + (dbinx)*binxo)

  // Process pixels in the intersection of the image rectangle
  // (1,1)-(M-1,N-1) and the keypoint bounding box.
  for(dyi =  VL_MAX (- W, 1 - yi    ) ;
      dyi <= VL_MIN (+ W, h - yi - 2) ; ++ dyi) {

    for(dxi =  VL_MAX (- W, 1 - xi    ) ;
        dxi <= VL_MIN (+ W, w - xi - 2) ; ++ dxi) {
      // retrieve
      float mod   = *( pt + dxi*xo + dyi*yo + 0 ) ;
      float angle = *( pt + dxi*xo + dyi*yo + 1 ) ;
      float theta = vl_mod_2pi_f (angle - angle0) ;

      // fractional displacement
      float dx = xi + dxi - x;
      float dy = yi + dyi - y;

      // get the displacement normalized w.r.t. the keypoint
      //   orientation and extension
      float nx = ( ct0 * dx + st0 * dy) / SBP ;
      float ny = (-st0 * dx + ct0 * dy) / SBP ;
      float nt = NBO * theta / (2 * M_PI) ;

      // Get the Gaussian weight of the sample. The Gaussian window
      // has a standard deviation equal to NBP/2. Note that dx and dy
      // are in the normalized frame, so that -NBP/2 <= dx <=
      // NBP/2. 
      float const wsigma = WINDOW_SIZE;
      float win = exp(-(nx*nx + ny*ny)/(2.0 * wsigma * wsigma)) ;

      // The sample will be distributed in 8 adjacent bins.
      //   We start from the ``lower-left'' bin.
      int         binx = (int)floor (nx - 0.5) ;
      int         biny = (int)floor (ny - 0.5) ;
      int         bint = (int)floor (nt) ;
      float rbinx = nx - (binx + 0.5) ;
      float rbiny = ny - (biny + 0.5) ;
      float rbint = nt - bint ;
      int         dbinx ;
      int         dbiny ;
      int         dbint ;

      // Distribute the current sample into the 8 adjacent bins
      for(dbinx = 0 ; dbinx < 2 ; ++dbinx) {
        for(dbiny = 0 ; dbiny < 2 ; ++dbiny) {
          for(dbint = 0 ; dbint < 2 ; ++dbint) {

            if (binx + dbinx >= - (NBP/2) &&
                binx + dbinx <    (NBP/2) &&
                biny + dbiny >= - (NBP/2) &&
                biny + dbiny <    (NBP/2) ) {
              float weight = win
                * mod
                * vl_abs_f (1 - dbinx - rbinx)
                * vl_abs_f (1 - dbiny - rbiny)
                * vl_abs_f (1 - dbint - rbint) ;

              atd(binx+dbinx, biny+dbiny, (bint+dbint) % NBO) += weight ;
            }
          }
        }
      }
    }
  }

  // Standard SIFT descriptors are normalized, truncated and normalized again
  {
    // normalize L2 norm
    vl_normalize_histogram (descr, descr + NBO*NBP*NBP) ;
    // truncate at 0.2.
    for(bin = 0; bin < NBO*NBP*NBP ; ++ bin) {
      if (descr [bin] > 0.2) descr [bin] = 0.2;
    }
    // normalize again. 
    vl_normalize_histogram (descr, descr + NBO*NBP*NBP) ;
  }
}

BOOST_FIXTURE_TEST_SUITE( test_setup, T )

BOOST_AUTO_TEST_CASE( test_sift_vlfeat_comparison )
{
  // Generates random image
  blitz::Array<double,2> A(HEIGHT,WIDTH);
  ranlib::Uniform<double> gen;
  for (int i=0; i<HEIGHT; ++i)
    for (int j=0; j<WIDTH; ++j)
      A(i,j) = gen.random();

  // 1. bob version used the friend test class
  blitz::Array<double,3> bob_descr(NBP,NBP,NBO);
  bob::ip::SIFTtest bob_test;
  bob_test.run(A, bob_descr);

  // 2. VLfeat version
  blitz::Array<float,3> vl_grad(HEIGHT,WIDTH,2);
  blitz::Array<float,3> vl_descr(NBP,NBP,NBO);
  vl_sift(A, vl_grad, vl_descr);

  // 3. Compare
  checkBlitzClose( bob_descr, vl_descr, eps); 
}

BOOST_AUTO_TEST_SUITE_END()
