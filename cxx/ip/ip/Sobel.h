/**
 * @file cxx/ip/ip/Sobel.h
 * @date Fri Apr 29 12:13:22 2011 +0200
 * @author Laurent El Shafey <Laurent.El-Shafey@idiap.ch>
 *
 * @brief This file provides a class to process images with the Sobel operator
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

#ifndef BOB_IP_SOBEL_H
#define BOB_IP_SOBEL_H

#include "core/array_assert.h"
#include "core/cast.h"
#include "ip/Exception.h"
#include "sp/conv.h"
#include "sp/extrapolate.h"

namespace bob {
/**
 * \ingroup libip_api
 * @{
 *
 */
  namespace ip {

  /**
   * @brief This class can be used to process images with the Sobel operator
  */
  class Sobel
  {
    public:

      /**
        * @brief Constructor: generates the Sobel kernel
        */
      Sobel(const bool up_positive=false, const bool left_positive=false, 
        const bob::sp::Conv::SizeOption size_opt=bob::sp::Conv::Same,
        const bob::sp::Extrapolation::BorderType 
          border_type=bob::sp::Extrapolation::Mirror);

      /**
        * @brief Destructor
        */
      virtual ~Sobel() {}

      /**
        * @brief Process a 2D blitz Array/Image by applying the Sobel operator
        *   The resulting 3D array will contain two planes:
        *     - The first one for the convolution with the y-kernel
        *     - The second one for the convolution with the x-kernel
        * @warning The selected type should be signed (e.g. int64_t or double)
        */
      template <typename T> void operator()(const blitz::Array<T,2>& src, 
        blitz::Array<T,3>& dst);

    private:
      /**
        * @brief Generates the Sobel kernels
        */
      void computeKernels();

      // Attributes
      blitz::Array<double, 2> m_kernel_y;
      blitz::Array<double, 2> m_kernel_x;
      bool m_up_positive;
      bool m_left_positive;
      bob::sp::Conv::SizeOption m_size_opt;
      bob::sp::Extrapolation::BorderType m_border_type;
  };

  template <typename T> 
  void Sobel::operator()(const blitz::Array<T,2>& src, blitz::Array<T,3>& dst)
  { 
    // Check that dst has two planes
    if( dst.extent(0) != 2 )
      throw bob::ip::Exception();
    // Check that dst has zero bases
    bob::core::array::assertZeroBase(dst);

    // Define slices for y and x
    blitz::Array<T,2> dst_y = dst(0, blitz::Range::all(), blitz::Range::all());
    blitz::Array<T,2> dst_x = dst(1, blitz::Range::all(), blitz::Range::all());
    // TODO: improve the way we deal with types
    if(m_border_type == bob::sp::Extrapolation::Zero || m_size_opt == bob::sp::Conv::Valid)
    {
     bob::sp::conv(src, bob::core::cast<T>(m_kernel_y), dst_y, m_size_opt);
     bob::sp::conv(src, bob::core::cast<T>(m_kernel_x), dst_x, m_size_opt);
    }
    else
    {
      blitz::Array<T,2> tmpy(bob::sp::getConvOutputSize(src, bob::core::cast<T>(m_kernel_y), bob::sp::Conv::Full));
      blitz::Array<T,2> tmpx(bob::sp::getConvOutputSize(src, bob::core::cast<T>(m_kernel_x), bob::sp::Conv::Full));
      if(m_border_type == bob::sp::Extrapolation::NearestNeighbour) {
        bob::sp::extrapolateNearest(src, tmpy);
        bob::sp::extrapolateNearest(src, tmpx);
      }
      else if(m_border_type == bob::sp::Extrapolation::Circular) {
        bob::sp::extrapolateCircular(src, tmpy);
        bob::sp::extrapolateCircular(src, tmpx);
      }
      else if(m_border_type == bob::sp::Extrapolation::Mirror) {
        bob::sp::extrapolateMirror(src, tmpy);
        bob::sp::extrapolateMirror(src, tmpx);
      }
      bob::sp::conv(tmpy, bob::core::cast<T>(m_kernel_y), dst_y, bob::sp::Conv::Valid);
      bob::sp::conv(tmpx, bob::core::cast<T>(m_kernel_x), dst_x, bob::sp::Conv::Valid);
    }
  }

}}

#endif /* BOB_SOBEL_H */
