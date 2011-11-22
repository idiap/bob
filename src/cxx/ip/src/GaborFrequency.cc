/**
 * @file cxx/ip/src/GaborFrequency.cc
 * @date Wed Apr 13 20:12:03 2011 +0200
 * @author Laurent El Shafey <Laurent.El-Shafey@idiap.ch>
 *
 * @brief This file provides a class to perform a Gabor filtering in the
 * frequency domain.
 *
 * Copyright (C) 2011 Idiap Reasearch Institute, Martigny, Switzerland
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

#include "core/array_assert.h"
#include "ip/GaborFrequency.h"
#include "sp/FFT2D.h"
#include "sp/fftshift.h"
#include "math/norminv.h"

namespace ip = Torch::ip;
namespace tca = Torch::core::array;

ip::GaborFrequency::GaborFrequency( const int height, const int width,
  const double f, const double theta,  const double gamma, const double eta, 
  const double pf, const bool cancel_dc, const bool use_envelope, 
  const bool output_in_frequency):
  m_height(height), m_width(width), m_f(f), m_theta(theta), m_gamma(gamma),
  m_eta(eta), m_pf(pf), m_cancel_dc(cancel_dc), m_use_envelope(use_envelope),
  m_output_in_frequency(output_in_frequency),
  m_kernel_shifted(height,width), m_kernel(height,width),
  m_work1(height,width), m_work2(height,width), 
  m_fft(new Torch::sp::FFT2D(height,width)), 
  m_ifft(new Torch::sp::IFFT2D(height,width))
{
  computeFilter();
  initWorkArrays();
}

ip::GaborFrequency::GaborFrequency( const ip::GaborFrequency& other):
  m_height(other.m_height), m_width(other.m_width), m_f(other.m_f), m_theta(other.m_theta), m_gamma(other.m_gamma),
  m_eta(other.m_eta), m_pf(other.m_pf), m_cancel_dc(other.m_cancel_dc), m_use_envelope(other.m_use_envelope),
  m_output_in_frequency(other.m_output_in_frequency),
  m_kernel_shifted(Torch::core::array::ccopy(other.m_kernel_shifted)),
  m_kernel(Torch::core::array::ccopy(other.m_kernel)),
  m_env_height(other.m_env_height), m_env_width(other.m_env_width),
  m_env_y_min(other.m_env_y_min), m_env_y_max(other.m_env_y_max),
  m_env_x_min(other.m_env_x_min), m_env_x_max(other.m_env_x_max),
  m_env_y_offset(other.m_env_y_offset), m_env_x_offset(other.m_env_x_offset),
  m_fft(new Torch::sp::FFT2D(other.m_height,other.m_width)), 
  m_ifft(new Torch::sp::IFFT2D(other.m_height,other.m_width))
{
  initWorkArrays();
}

ip::GaborFrequency::~GaborFrequency() 
{ 
}

void ip::GaborFrequency::operator()( 
  const blitz::Array<std::complex<double>,2>& src,
  blitz::Array<std::complex<double>,2>& dst)
{ 
  // Checks input
  tca::assertZeroBase(src);
  tca::assertSameShape(src,m_kernel_shifted);
  // Checks output
  tca::assertZeroBase(dst);
  tca::assertSameShape(dst,m_kernel_shifted);

  // Filters in the frequency domain
  if( !m_use_envelope)
  {
    // 1/ Computes FFT
    blitz::Array<std::complex<double>,2> src_fft(src.shape());
    m_fft->operator()( src, src_fft);
    // 2/ Filters in the frequency domain (elementwise multiplication)
    m_work1 = src_fft * m_kernel;
    // 3/ Output back in the spatial domain (IFFT)
    m_ifft->operator()( m_work1, dst);
  }
  else
  {
    // 1/ Computes FFT
    m_fft->operator()( src, m_work1);
    Torch::sp::fftshift<std::complex<double> >( m_work1, m_work2); // m_work2 <-> src_fft_shift

    // 2/ Filters in the frequency domain 
    //    (elementwise multiplication of the non-zero area)
    m_work1 = 0.; // m_work1 will contain dst_fft_shift
    // Determines the 'valid' non-zero range using the envelope offset/size
    blitz::Range k_h(m_env_y_offset, m_env_y_offset + m_env_height-1);
    blitz::Range k_w(m_env_x_offset, m_env_x_offset + m_env_width-1);
    blitz::Array<std::complex<double>,2> work1_s = m_work1(k_h,k_w);
    blitz::Array<std::complex<double>,2> work2_s = m_work2(k_h,k_w);
    blitz::Array<std::complex<double>,2> kernel_shifted_s = m_kernel_shifted(k_h,k_w);
    // Filters in the Fourier domain
    work1_s = work2_s * kernel_shifted_s;

    // 3/ Output back in the spatial domain (IFFT)
    Torch::sp::ifftshift<std::complex<double> >( m_work1, m_work2); // m_work2 <-> dst_fft
    m_ifft->operator()( m_work2, dst);
  }
}

void ip::GaborFrequency::initWorkArrays()
{
  m_work1.resize( m_height, m_width);
  m_work2.resize( m_height, m_width);
}

void ip::GaborFrequency::computeFilter()
{
  // Computes some constant values used later
  const double pi2 = M_PI*M_PI;
  const double cos_theta = cos(m_theta);
  const double sin_theta = sin(m_theta);
  const double gamma2 = m_gamma * m_gamma;
  const double eta2 = m_eta * m_eta;
  const double f2 = m_f*m_f;
  const double pi2_f2 = pi2 / f2;
  const double exp_m_pi2_gamma2 = exp(-pi2 * gamma2);
  const double width_d = m_width;
  const double height_d = m_height;

  // Declares variable to handle both cases (with/without envelope)
  // and thus factorize the code
  int h_offset;
  int w_offset;
  blitz::Array<std::complex<double>, 2> kernel_shifted_slice; 

  if( !m_use_envelope)
  {
    // Resizes the frequency filter
    m_kernel.resize( m_height, m_width );
    m_kernel_shifted.resize( m_height, m_width );

    // Defines the offset values
    h_offset = -( m_height / 2 - (m_height % 2 == 1 ? 0 : 1) );
    w_offset = -( m_width / 2 - (m_width % 2 == 1 ? 0 : 1) );
    kernel_shifted_slice.reference( m_kernel_shifted );
  }
  else
  {
    // Computes the envelope given, the length of the major and minor axes
    computeEnvelope();

    // Resizes the frequency filter
    m_kernel.resize( m_height, m_width );
    m_kernel_shifted.resize( m_height, m_width );

    // Initializes to zero 
    m_kernel_shifted = 0.;

    // Defines the offset values
    h_offset = m_env_y_min;
    w_offset = m_env_x_min;

    // Reference to the slice of the filter which has non-zero values
    kernel_shifted_slice.reference(
      m_kernel_shifted( blitz::Range(m_env_y_offset, m_env_y_offset + m_env_height-1),
        blitz::Range(m_env_x_offset, m_env_x_offset + m_env_width-1) ) );
  }

  // Computes the filter
  // G(v,u) = exp( -pi**2/f**2 ( (u'-f)**2/alpha**2 - v'**2/eta**2) ) 
  // where u' = u*cos(theta) + v*sin(theta) and v' = -u*sin(theta) + v*cos(theta)
  blitz::firstIndex y;
  blitz::secondIndex x;
  kernel_shifted_slice = exp( -pi2_f2 * 
      ( gamma2 * 
        ( (x+w_offset)/width_d * cos_theta + (y+h_offset)/height_d * sin_theta - m_f) * 
        ( (x+w_offset)/width_d * cos_theta + (y+h_offset)/height_d * sin_theta - m_f) + 
        eta2 * 
        ( -(x+w_offset)/width_d * sin_theta + (y+h_offset)/height_d * cos_theta) * 
        ( -(x+w_offset)/width_d * sin_theta + (y+h_offset)/height_d * cos_theta) 
      ) 
    );
    

  // Removes DC component if required
  // G(v,u) -= exp( -pi**2 * gamma**2 ) * 
  //             exp( -pi**2/f**2 * (gamma**2*u'**2  + eta**2*v'**2) ) 
  // where u' = u*cos(theta) + v*sin(theta) and v' = -u*sin(theta) + v*cos(theta)
  if( m_cancel_dc ) 
  {
    kernel_shifted_slice -= exp_m_pi2_gamma2 * 
      exp( -pi2_f2 * 
        ( gamma2 * 
          ( (x+w_offset)/width_d * cos_theta + (y+h_offset)/height_d * sin_theta ) * 
          ( (x+w_offset)/width_d * cos_theta + (y+h_offset)/height_d * sin_theta ) +
        eta2 * 
          ( -(x+w_offset)/width_d * sin_theta + (y+h_offset)/height_d * cos_theta) * 
          ( -(x+w_offset)/width_d * sin_theta + (y+h_offset)/height_d * cos_theta) 
        ) 
      );
  }

  // Computes the non_shifted version
  Torch::sp::ifftshift<std::complex<double> >( m_kernel_shifted, m_kernel );
}

// TODO: Needs to be refactored: Geometry module/functions: Rotation, ellipsoid
void ip::GaborFrequency::computeEnvelope()
{
  // Computes variable
  double sq_pf=sqrt(m_pf);

  // Defines the envelope
  double major_env_max = Torch::math::norminv((1.+sq_pf)/2., 0., m_f/m_gamma/(sqrt(2) * M_PI) );
  double minor_env_max = Torch::math::norminv((1.+sq_pf)/2., 0., m_f/m_eta/(sqrt(2) * M_PI) );

  // doubles which absolute values are below EPSILON are considered to be equal to zero
  double EPSILON = 1000 * std::numeric_limits<double>::epsilon();

  // Temporary variables
  double points[8];
  double x1, y1, x2, y2;

  // Tests if theta = n * PI/2, n integer 
  // (particular case to avoid infinite slopes)
  int theta_mod = m_theta - static_cast<int>( m_theta / M_PI_2 ) * M_PI_2;
  if ( fabs(theta_mod) < EPSILON )
  {
    x1=-major_env_max;
    y1=0;
    x2=0;
    y2=minor_env_max;
  }
  else
  {
    // Finds the points such that the slope of the tangent at this point is:
    // -tan(pi/2-theta) or tan(theta)
    double tn_pi2_theta = -tan(M_PI_2 - m_theta);
    double tn_theta = tan(m_theta);
    double a2 = major_env_max*major_env_max;
    double b2 = minor_env_max*minor_env_max;

    x1 = (tn_pi2_theta * a2) / sqrt(b2 + tn_pi2_theta*tn_pi2_theta*a2);
    y1 = minor_env_max / major_env_max * sqrt(a2 - x1*x1);
    x2 = (tn_theta * a2) / sqrt(b2 + tn_theta*tn_theta*a2);
    y2 = minor_env_max / major_env_max * sqrt(a2 - x2*x2);
  }

  // Updates the value of the points
  points[0]=x1+m_f;
  points[1]=y1;
  points[2]=-x1+m_f;
  points[3]=-y1;
  points[4]=x2+m_f;
  points[5]=y2;
  points[6]=-x2+m_f;
  points[7]=-y2;

  // Rotates the points by theta to get the final envelope
  double cos_theta=cos(m_theta);
  double sin_theta=sin(m_theta);

  // Temporary variable to store the point value we have just erased
  double p_svg;

  // Rotates the points with a m_theta angle
  for(int i=0;i<4;i++)
  {
    p_svg=points[2*i];
    points[2*i]=points[2*i]*cos_theta-points[2*i+1]*sin_theta;
    points[2*i+1]=p_svg*sin_theta+points[2*i+1]*cos_theta;
  }

  // Finds minima and maxima
  double env[4];
  env[0] = std::numeric_limits<int>::max(); // x_min
  env[1] = std::numeric_limits<int>::min(); // x_max
  env[2] = std::numeric_limits<int>::max(); // y_min
  env[3] = std::numeric_limits<int>::min(); // y_max
  for(int i=0;i<4;i++)
  {
    // Updates x_min and x_max
    if(points[2*i]<env[0]) env[0]=points[2*i];
    if(points[2*i]>env[1]) env[1]=points[2*i];

    // Updates y_min and y_max
    if(points[2*i+1]<env[2]) env[2]=points[2*i+1];
    if(points[2*i+1]>env[3]) env[3]=points[2*i+1];
  }
  // Coordinates of the envelope 
  // (can be negative as they are relative to the frequency origin)
  m_env_y_min = floor(env[2] * m_height);
  m_env_y_max = ceil(env[3]  * m_height);
  m_env_x_min = floor(env[0] * m_width);
  m_env_x_max = ceil(env[1]  * m_width);
  // Size of the envelope 
  m_env_height = m_env_y_max - m_env_y_min + 1;
  m_env_width = m_env_x_max - m_env_x_min + 1;
  // Computes envelope offset wrt. the full filter in the frequency domain
  m_env_y_offset = m_height / 2 - (m_height % 2 == 1 ? 0 : 1) + m_env_y_min;
  m_env_x_offset = m_width / 2 - (m_width % 2 == 1 ? 0 : 1) + m_env_x_min;

  // Checks that the envelope is not outside the input image/window
  if(m_env_y_offset < 0) m_env_y_offset = 0;
  if(m_env_y_offset >= m_height) m_env_y_offset = m_height-1;
  if(m_env_x_offset < 0) m_env_x_offset = 0;
  if(m_env_x_offset >= m_width) m_env_x_offset = m_width-1;
  if(m_env_y_offset + m_env_height < 0) m_env_height = 0;
  if(m_env_y_offset + m_env_height >= m_height) m_env_height = m_height-1 - m_env_y_offset;
  if(m_env_x_offset + m_env_width < 0) m_env_width = 0;
  if(m_env_x_offset + m_env_width >= m_width) m_env_width = m_width-1 - m_env_x_offset;
}
