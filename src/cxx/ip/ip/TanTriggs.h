/**
 * @file cxx/ip/ip/TanTriggs.h
 * @date Fri Mar 18 18:09:25 2011 +0100
 * @author Laurent El Shafey <Laurent.El-Shafey@idiap.ch>
 *
 * @brief This file provides a class to perform Tan and Triggs preprocessing.
 *
 * Copyright (C) 2011-2012 Idiap Reasearch Institute, Martigny, Switzerland
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

#ifndef BOB5SPRO_TAN_TRIGGS_H
#define BOB5SPRO_TAN_TRIGGS_H 1

#include "core/logging.h"
#include "ip/Exception.h"
#include "ip/gammaCorrection.h"
#include "sp/convolution.h"

namespace bob {
/**
 * \ingroup libip_api
 * @{
 *
 */
  namespace ip {

  /**
   * @brief This class can be used to perform Tan and Triggs preprocessing.
   *   This algorithm is described in the following articles:
   *    1) "Enhanced Local Texture Feature Sets for Face Recognition Under
   *         Difficult Lighting Conditions", from Xiaoyang Tan and Bill Triggs,
   *       in the proceedings of IEEE International Workshop on Analysis and 
   *       Modelling of Faces and Gestures (AMFG), 2007, p. 162-182.
   *    2) "Enhanced Local Texture Feature Sets for Face Recognition Under 
   *         Difficult Lighting Conditions", from Xiaoyang Tan and Bill Triggs,
   *       in IEEE Transactions on Image Processing, June 2010, Issue 6, 
   *       Volume 19, p. 1635-1650.
   *       (http://ieeexplore.ieee.org/xpl/freeabs_all.jsp?arnumber=5411802)
  */
	class TanTriggs
	{
  	public:

	  	/**
        * @brief Constructor: generates the Difference of Gaussians filter
        */
	    TanTriggs(const double gamma=0.2, const double sigma0=1., 
        const double sigma1=2., const int size=2, const double threshold=10., 
        const double alpha=0.1, const enum sp::Convolution::SizeOption 
        size_opt=sp::Convolution::Same, const enum sp::Convolution::BorderOption 
        border_opt=sp::Convolution::Mirror);

	  	/**
        * @brief Destructor
        */
	    virtual ~TanTriggs();

	  	/**
        * @brief Process a 2D blitz Array/Image by applying the preprocessing
        * algorihtm
        */
	    template <typename T> void operator()(const blitz::Array<T,2>& src, 
        blitz::Array<double,2>& dst);

	  private:
	  	/**
        * @brief Perform the contrast equalization step on a 2D blitz 
        * Array/Image.
        */
      void performContrastEqualization( blitz::Array<double,2>& img);

	  	/**
        * @brief Generate the difference of Gaussian filter
        */
  		void computeDoG(double sigma0, double sigma1, int size);

      // Attributes
      blitz::Array<double, 2> m_kernel;
      blitz::Array<double, 2> m_img_tmp;
      double m_gamma;
      double m_sigma0;
      double m_sigma1;
      int m_size;
      double m_threshold;
      double m_alpha;
      enum sp::Convolution::SizeOption m_size_opt;
      enum sp::Convolution::BorderOption m_border_opt;
	};

  template <typename T> 
  void TanTriggs::operator()(const blitz::Array<T,2>& src, 
    blitz::Array<double,2>& dst) 
  { 
    // Check and reindex if required
    if( dst.base(0) != 0 || dst.base(1) != 0 ) { 
      const blitz::TinyVector<int,2> zero_base = 0;
      dst.reindexSelf( zero_base );
    }
    // Check and resize dst if required
    if( dst.extent(0) != src.extent(0) || dst.extent(1) != src.extent(1) )
      dst.resize( src.extent(0), src.extent(1) );

    // Check and resize intermediate array if required
    if( m_img_tmp.extent(0) != src.extent(0) ||  
      m_img_tmp.extent(1) != src.extent(1) )
      m_img_tmp.resize( src.extent(0), src.extent(1) );

    // 1/ Perform gamma correction
    if( m_gamma > 0.)
      bob::ip::gammaCorrection( src, m_img_tmp, m_gamma);
    else
    {
      blitz::Range src_y( src.lbound(0), src.ubound(0)),
                   src_x( src.lbound(1), src.ubound(1));
      blitz::Range tmp_y( m_img_tmp.lbound(0), m_img_tmp.ubound(0)),
                   tmp_x( m_img_tmp.lbound(1), m_img_tmp.ubound(1));
      m_img_tmp(tmp_y,tmp_x) = log( 1. + src(src_y,src_x) );
    }

    // 2/ Convolution with the DoG Filter
    bob::sp::convolve( m_img_tmp, m_kernel, dst, 
      m_size_opt, m_border_opt);

    // 3/ Perform contrast equalization
    performContrastEqualization(dst);
  }

}}

#endif /* BOB5SPRO_TAN_TRIGGS_H */
