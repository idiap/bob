/**
 * @file cxx/ip/ip/VLSIFT.h
 * @date Sun Dec 18 2011
 * @author Laurent El Shafey <Laurent.El-Shafey@idiap.ch>
 *
 * @brief This file defines classes to compute SIFT features using VLFeat
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

#ifndef TORCH5SPRO_IP_VLSIFT_H
#define TORCH5SPRO_IP_VLSIFT_H

#include <blitz/array.h>
#include <stdint.h> // uint16_t declaration
#include <vector>
#include <boost/shared_ptr.hpp>
#include <vl/generic.h>
#include <vl/sift.h>

namespace Torch {
/**
 * \ingroup libip_api
 * @{
 *
 */
  namespace ip {

    /**
      * @brief This class allows the computation of SIFT features
      *   For more information, please refer to the following article:
      *     "Distinctive Image Features from Scale-Invariant Keypoints", 
      *     from D.G. Lowe,
      *     International Journal of Computer Vision, 60, 2, pp. 91-110, 2004
      */
    class VLSIFT
    {
      public:
        /**
          * @brief Constructor
          */
        VLSIFT(const int height, const int width, const int n_intervals,
          const int n_octaves, const int octave_min,
          const double peak_thres=0.03, const double edge_thres=10, const double magnif=3);

        /**
          * @brief Destructor
          */
        virtual ~VLSIFT();

        /**
          * @brief Extract SIFT features from a 2D blitz::Array, and save 
          *   the resulting features in the dst vector of 1D blitz::Arrays.
          */
        void operator()(const blitz::Array<uint8_t,2>& src, 
          std::vector<blitz::Array<double,1> >& dst);


    	protected:
        /**
          * @brief Attributes
          */
        int m_height;
        int m_width;
        int m_n_intervals;
        int m_n_octaves;
        int m_octave_min;
        double m_peak_thres;
        double m_edge_thres;
        double m_magnif;
        double m_sigma;
        double m_sigma_init;
        int m_kernel_size;
        VlSiftFilt *m_filt;
        vl_uint8 *m_data;
        vl_sift_pix *m_fdata;

        bool m_firstoct_conv_req; // indicates if the first scale of the first octave requires a convolution
    };

  }
}

#endif /* TORCH5SPRO_IP_VLSIFT_H */
