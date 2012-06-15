/**
 * @file cxx/ip/ip/LBPTopOperator.h
 * @date Tue Apr 26 19:20:57 2011 +0200
 * @author Laurent El Shafey <Laurent.El-Shafey@idiap.ch>
 *
 * @brief This class can be used to calculate the LBP-Top  of a set of image
 * frames representing a video sequence (c.f. Dynamic Texture
 * Recognition Using Local Binary Patterns with an Application to Facial
 * Expression from Zhao & Pietikäinen, IEEE Trans. on PAMI, 2007)
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

#ifndef BOB_IP_LBPTOPOPERATOR_H 
#define BOB_IP_LBPTOPOPERATOR_H

#include <blitz/array.h>
#include <algorithm>
#include "ip/LBP.h"
#include "ip/LBP4R.h"
#include "ip/LBP8R.h"
#include "ip/Exception.h"
#include <boost/shared_ptr.hpp>

namespace bob {
/**
 * \ingroup libip_api
 * @{
 *
 */
  namespace ip {
    /**
     * The LBPTopOperator class is designed to calculate the LBP-Top
     * coefficients given a set of images. 
     *
     * The workflow is as follows:
     * TODO: UPDATE as this is not true
     * 1. You initialize the class, defining the radius and number of points
     * in each of the three directions: XY, XT, YT for the LBP calculations
     * 2. For each image you have in the frame sequence, you push into the 
     * class
     * 3. An internal FIFO queue (length = radius in T direction) keeps track 
     * of the current image and their order. As a new image is pushed in, the
     * oldest on the queue is pushed out. 
     * 4. After pushing an image, you read the current LBP-Top coefficients
     * and may save it somewhere.
     */
    class LBPTopOperator
    {
      public:
        /**
         * Constructs a new LBPTopOperator object starting from the algorithm
         * configuration. Please note this object will always produce rotation
         * invariant 2D codes, also taking into consideration pattern
         * uniformity (u2 variant). 
         *
         * The radius in X (width) direction is combied with the radius in the
         * Y (height) direction for the calculation of the LBP on the XY 
         * (frame) direction. The radius in T is taken from the number of 
         * frames input, so it is dependent on the input to 
         * LBPTopOperator:operator().
         *
         * @warning The current number of points supported in bob is either
         * 8 or 4. Any values differing from that need implementation of 
         * specialized functionality.
         *
         * @param radius_xy The radius to be used at the XY plane
         * @param points_xy The number of points to use for the calculation of
         * the 2D LBP on the XY plane (frame)
         * @param radius_xt The radius to be used at the XT plane
         * @param points_xt The number of points to use for the calculation of
         * the 2D LBP on the XT plane
         * @param radius_yt The radius to be used at the YT plane
         * @param points_yt The number of points to use for the calculation of
         * the 2D LBP on the YT plane
         */
        LBPTopOperator(int radius_xy, int points_xy, int radius_xt, 
          int points_xt, int radius_yt, int points_yt);

        /**
         * Destructor
         */
        virtual ~LBPTopOperator() {}

        /**
         * Processes a 3D array representing a set of <b>grayscale</b> images 
         * and returns (by argument) the three LBP planes calculated. The 3D 
         * array has to be arranged in this way:
         *
         * 1st dimension => time
         * 2nd dimension => frame height
         * 3rd dimension => frame width
         *
         * TODO: The following is not true anymore! 
         *       The three planes now intersect at the central pixel
         * The number of frames in the array has to be always an odd number.
         * The central frame is taken as the frame where the LBP planes have
         * to be calculated from. The radius in dimension T (3rd dimension) is
         * taken to be (N-1)/2 where N is the number of frames input.
         *
         * @param src The input 3D array as described in the documentation
         * of this method.
         * @param xy The result of the LBP operator in the XY plane (frame),
         * for the central frame of the input array. This is an image.
         * @param xt The result of the LBP operator in the XT plane for the
         * whole image, taking into consideration the size of the width of the
         * input array along the time direction. 
         * @param yt The result of the LBP operator in the YT plane for the
         * whole image, taking into consideration the size of the width of the
         * input array along the time direction. 
         *
         */
        void operator()(const blitz::Array<uint8_t,3>& src, 
          blitz::Array<uint16_t,2>& xy,
          blitz::Array<uint16_t,2>& xt,
          blitz::Array<uint16_t,2>& yt) const
        { process<uint8_t>( src, xy, xt, yt); }
        void operator()(const blitz::Array<uint16_t,3>& src, 
          blitz::Array<uint16_t,2>& xy,
          blitz::Array<uint16_t,2>& xt,
          blitz::Array<uint16_t,2>& yt) const
        { process<uint16_t>( src, xy, xt, yt); }
        void operator()(const blitz::Array<double,3>& src, 
          blitz::Array<uint16_t,2>& xy,
          blitz::Array<uint16_t,2>& xt,
          blitz::Array<uint16_t,2>& yt) const
        { process<double>( src, xy, xt, yt); }

      private:
        /**
          * Processes a 3D array representing a set of <b>grayscale</b> images
          * and returns (by argument) the three LBP planes calculated.
          */
        template <typename T> 
        void process(const blitz::Array<T,3>& src, 
          blitz::Array<uint16_t,2>& xy,
          blitz::Array<uint16_t,2>& xt,
          blitz::Array<uint16_t,2>& yt) const;

        int m_radius_xy; ///< The LBPu2,i radius in XY
        int m_points_xy; ///< The number of points in the XY LBPu2,i
        int m_radius_xt; ///< The LBPu2,i radius in XT
        int m_points_xt; ///< The number of points in the XT LBPu2,i
        int m_radius_yt; ///< The LBPu2,i radius in YT
        int m_points_yt; ///< The number of points in the YT LBPu2,i
        boost::shared_ptr<bob::ip::LBP> m_lbp_xy; ///< The operator for the XY calculation
        boost::shared_ptr<bob::ip::LBP> m_lbp_xt; ///< The operator for the XT calculation
        boost::shared_ptr<bob::ip::LBP> m_lbp_yt; ///< The operator for the YT calculation
    };

    template <typename T>
    void bob::ip::LBPTopOperator::process(const blitz::Array<T,3>& src,
      blitz::Array<uint16_t,2>& xy,
      blitz::Array<uint16_t,2>& xt,
      blitz::Array<uint16_t,2>& yt) const
    {
      // TODO
      // Assert on input and output dimensions
      /*
      // we need an odd number, at (2N+1), where N = std::max(radius_xt, radius_yt)
      if(src.extent(2)%2 == 0) {
        // bob::warning("Cannot process a even-numbered set of frames");
        // TODO
        throw bob::ip::Exception();
      }
      
      const int N = std::max(m_radius_xt, m_radius_yt);
      if(src.extent(2) != (2*N+1) ) {
        // bob::warning("The number of input frames should be %d", 2*N+1);
        // TODO
        throw bob::ip::Exception();
      }
      */
      int Tlength = src.extent(0);
      int height = src.extent(1);
      int width = src.extent(2);
      int tc = Tlength/2;
      int yc = height/2;
      int xc = width/2;

      // XY plane calculation
      const blitz::Array<T,2> kxy = 
        src( tc, blitz::Range::all(), blitz::Range::all());
      //k.select(&tensor, 3, 2*N);
      //const int max_lbp_xy = m_lbp_xy->getMaxLabel();
      //const float inv_max_lbp_xy = 255.0f / (max_lbp_xy + 0.0f);
      for (int y=m_radius_xy; y < (height-m_radius_xy); ++y) {
        for (int x=m_radius_xy; x < (width-m_radius_xy); ++x) {
          //m_lbp_xy->setXY(x, y);
          xy(y-m_radius_xy,x-m_radius_xy) = m_lbp_xy->operator()(kxy, y, x);
          //xy(y,x) = static_cast<uint16_t>(  
          //  floor(m_lbp_xy->operator()(k, y, x) * inv_max_lbp_xy + 0.5));
          //xy.set(y, x, 0, (short)(inv_max_lbp_xy * m_lbp_xy->getLBP() + 0.5f));
        }
      }

      // XT plane calculation
      //const int max_lbp_xt = m_lbp_xt->getMaxLabel();
      //const float inv_max_lbp_xt = 255.0f / (max_lbp_xt + 0.0f);
      const blitz::Array<T,2> kxt = 
        src( blitz::Range::all(), yc, blitz::Range::all());
      for (int t = m_radius_xt; t < (Tlength-m_radius_xt); ++t) {
        //bob::ShortTensor k;
        //k.select(&tensor, 0, y);
        //bob::ShortTensor kt;
        //kt.transpose(&k, 1, 2); //get the gray levels on the last dimension
        for (int x=m_radius_xt; x < (width-m_radius_xt); ++x) {
          //m_lbp_xt->setXY(x, 2*N);
          //m_lbp_xt->process(kt);
          //xt.set(y, x, 0, (short)(inv_max_lbp_xt * m_lbp_xt->getLBP() + 0.5f));
          xt(t-m_radius_xt,x-m_radius_xt) = m_lbp_xt->operator()(kxt, t, x);
          //xt(y,x) = static_cast<uint16_t>(
          //   floor(m_lbp_xt->operator()(k, 2*N, x) * inv_max_lbp_xt + 0.5));
        }
      }

      // YT plane calculation
      //const int max_lbp_yt = m_lbp_yt->getMaxLabel();
      //const float inv_max_lbp_yt = 255.0f / (max_lbp_yt + 0.0f);
      const blitz::Array<T,2> kyt = 
        src( blitz::Range::all(), blitz::Range::all(), xc);
      for (int t = m_radius_yt; t < (Tlength-m_radius_yt); ++t) {
        //bob::ShortTensor k;
        //k.select(&tensor, 1, x);
        //bob::ShortTensor kt;
        //kt.transpose(&k, 1, 2); //get the gray levels on the last dimension
        for (int y = m_radius_yt; y < (height-m_radius_yt); ++y) {
          //m_lbp_yt->setXY(y, 2*N);
          //m_lbp_yt->process(kt);
          //yt.set(y, x, 0, (short)(inv_max_lbp_yt * m_lbp_yt->getLBP() + 0.5f));
          yt(t-m_radius_yt,y-m_radius_yt) = m_lbp_yt->operator()(kyt, t, y);
          //yt(y,x) = static_cast<uint16_t>(
          //   floor(m_lbp_yt->operator()(k, 2*N, y) * inv_max_lbp_yt + 0.5));
        }
      }
    }

  }
}

#endif /* BOB_IP_LBPTOPOPERATOR_H */
