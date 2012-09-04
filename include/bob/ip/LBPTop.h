/**
 * @file cxx/ip/ip/LBPTop.h
 * @date Tue Apr 26 19:20:57 2011 +0200
 * @author Laurent El Shafey <Laurent.El-Shafey@idiap.ch>
 * @author Tiago Freitas Pereira <Tiago.Pereira@idiap.ch>
 * @author Andre Anjos <andre.anjos@idiap.ch>
 *
 * This class can be used to calculate the LBP-Top  of a set of image frames
 * representing a video sequence (c.f. Dynamic Texture Recognition Using Local
 * Binary Patterns with an Application to Facial Expression from Zhao &
 * Pietikäinen, IEEE Trans. on PAMI, 2007)
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

#ifndef BOB_IP_LBPTOP_H 
#define BOB_IP_LBPTOP_H

#include <boost/shared_ptr.hpp>
#include <blitz/array.h>
#include <algorithm>
#include "ip/LBP.h"
#include "ip/LBP4R.h"
#include "ip/LBP8R.h"
#include "ip/Exception.h"

namespace bob { namespace ip {

  /**
   * The LBPTop class is designed to calculate the LBP-Top
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
  class LBPTop {

    public:

      /** 
       * Constructs a new LBPTop object starting from the algorithm
       * configuration
       *
       * @param lbp_xy The 2D LBP-XY plane configuration
       * @param lbp_xt The 2D LBP-XT plane configuration
       * @param lbp_yt The 2D LBP-YT plane configuration
       */
      LBPTop(const bob::ip::LBP& lbp_xy,
             const bob::ip::LBP& lbp_xt, 
             const bob::ip::LBP& lbp_yt);

      /**
       * Copy constructor
       */
      LBPTop(const LBPTop& other);

      /**
       * Destructor
       */
      virtual ~LBPTop();

      /**
       * Assignment
       */
      LBPTop& operator= (const LBPTop& other);

      /**
       * Processes a 3D array representing a set of <b>grayscale</b> images and
       * returns (by argument) the three LBP planes calculated. The 3D array
       * has to be arranged in this way:
       *
       * 1st dimension => time
       * 2nd dimension => frame height
       * 3rd dimension => frame width
       *
       * @param src The input 3D array as described in the documentation of
       * this method.
       * @param xy The result of the LBP operator in the XY plane (frame), for
       * the central frame of the input array. This is an image.
       * @param xt The result of the LBP operator in the XT plane for the whole
       * image, taking into consideration the size of the width of the input
       * array along the time direction. 
       * @param yt The result of the LBP operator in the YT plane for the whole
       * image, taking into consideration the size of the width of the input
       * array along the time direction. 
       */
      void operator()(const blitz::Array<uint8_t,3>& src, 
          blitz::Array<uint16_t,3>& xy,
          blitz::Array<uint16_t,3>& xt,
          blitz::Array<uint16_t,3>& yt) const;

      void operator()(const blitz::Array<uint16_t,3>& src, 
          blitz::Array<uint16_t,3>& xy,
          blitz::Array<uint16_t,3>& xt,
          blitz::Array<uint16_t,3>& yt) const;

      void operator()(const blitz::Array<double,3>& src, 
          blitz::Array<uint16_t,3>& xy,
          blitz::Array<uint16_t,3>& xt,
          blitz::Array<uint16_t,3>& yt) const;

      /**
       * Accessors
       */

      /**
       * Returns the XY plane LBP operator
       */
      const boost::shared_ptr<bob::ip::LBP> getXY() const { 
        return m_lbp_xy; 
      }

      /**
       * Returns the XT plane LBP operator
       */
      const boost::shared_ptr<bob::ip::LBP> getXT() const {
        return m_lbp_xt;
      }

      /**
       * Returns the YT plane LBP operator
       */
      const boost::shared_ptr<bob::ip::LBP> getYT() const {
        return m_lbp_yt;
      }

    private: //representation and methods

      /**
       * Processes a 3D array representing a set of <b>grayscale</b> images and
       * returns (by argument) the three LBP planes calculated.
       */
      template <typename T> 
        void process(const blitz::Array<T,3>& src, 
            blitz::Array<uint16_t,3>& xy,
            blitz::Array<uint16_t,3>& xt,
            blitz::Array<uint16_t,3>& yt) const;

      boost::shared_ptr<bob::ip::LBP> m_lbp_xy; ///< LBP for the XY calculation
      boost::shared_ptr<bob::ip::LBP> m_lbp_xt; ///< LBP for the XT calculation
      boost::shared_ptr<bob::ip::LBP> m_lbp_yt; ///< LBP for the YT calculation
  };

  /**
   * Implementation of certain template methods.
   */

  template <typename T>
    void bob::ip::LBPTop::process(const blitz::Array<T,3>& src,
                                  blitz::Array<uint16_t,3>& xy,
                                  blitz::Array<uint16_t,3>& xt,
                                  blitz::Array<uint16_t,3>& yt) const
    {


      int radius_x = m_lbp_xy->getRadius();  ///< The LBPu2,i radius in X direction
      int radius_y = m_lbp_xy->getRadius2(); ///< The LBPu2,i radius in Y direction
      int radius_t = m_lbp_yt->getRadius2();  ///< The LBPu2,i radius in T direction


      int Tlength = src.extent(0);
      int height = src.extent(1);
      int width = src.extent(2);


      /***** Checking the inputs *****/
      /**** Get XY plane (the first is enough) ****/

      const blitz::Array<T,2> checkXY = 
        src( 0, blitz::Range::all(), blitz::Range::all());
      m_lbp_xy->operator()(checkXY, radius_x, radius_y);

      /**** Get XT plane (Intersect in one point is enough) ****/
      int limitT = ceil(2*radius_t + 1);
      if( Tlength < limitT )
        throw ParamOutOfBoundaryError("t_radius", false, Tlength, limitT);


      /***** Checking the outputs *****/
      int max_radius = radius_x > radius_y ? radius_x : radius_y;
      max_radius = max_radius > radius_t ? max_radius : radius_t;
      int limitWidth  = width-2*max_radius;
      int limitHeight = height-2*max_radius;
      int limitTime   = Tlength-2*max_radius;

      /*Checking XY*/
      if( xy.extent(0) != limitTime)
        throw ParamOutOfBoundaryError("Time parameter in  XY ", (xy.extent(0) > limitTime), xy.extent(0), limitTime);
      if( xy.extent(1) != limitHeight)
        throw ParamOutOfBoundaryError("Height parameter in  XY ", (xy.extent(1) > limitWidth), xy.extent(1), limitWidth);
      if( xy.extent(2) != limitWidth)
        throw ParamOutOfBoundaryError("Width parameter in  XY ", (xy.extent(2) > limitHeight), xy.extent(2), limitHeight);

      /*Checking XT*/
      if( xt.extent(0) != limitTime)
        throw ParamOutOfBoundaryError("Time parameter in  XT ", (xt.extent(0) > limitTime), xt.extent(0), limitTime);
      if( xt.extent(1) != limitHeight)
        throw ParamOutOfBoundaryError("Height parameter in  XT ", (xt.extent(1) > limitWidth), xt.extent(1), limitWidth);
      if( xt.extent(2) != limitWidth)
        throw ParamOutOfBoundaryError("Width parameter in  XT ", (xt.extent(2) > limitHeight), xt.extent(2), limitHeight);

      /*Checking YT*/
      if( yt.extent(0) != limitTime)
        throw ParamOutOfBoundaryError("Time parameter in  YT ", (yt.extent(0) > limitTime), yt.extent(0), limitTime);
      if( yt.extent(1) != limitHeight)
        throw ParamOutOfBoundaryError("Height parameter in  YT ", (yt.extent(1) > limitWidth), yt.extent(1), limitWidth);
      if( yt.extent(2) != limitWidth)
        throw ParamOutOfBoundaryError("Width parameter in  YT ", (yt.extent(2) > limitHeight), yt.extent(2), limitHeight);


      //for each element in time domain (the simplest way to see what is happening)
      for(int i=max_radius;i<(Tlength-max_radius);i++){
        for (int j=max_radius; j < (height-max_radius); j++) {
          for (int k=max_radius; k < (width-max_radius); k++) {
            /*Getting the "micro-plane" for XY calculus*/
            const blitz::Array<T,2> kxy = 
               src( i, blitz::Range(j-max_radius,j+max_radius), blitz::Range(k-max_radius,k+max_radius));
            xy(i-max_radius,j-max_radius,k-max_radius) = m_lbp_xy->operator()(kxy, max_radius, max_radius);

            /*Getting the "micro-plane" for XT calculus*/
            const blitz::Array<T,2> kxt = 
               src(blitz::Range(i-max_radius,i+max_radius),j,blitz::Range(k-max_radius,k+max_radius));
            xt(i-max_radius,j-max_radius,k-max_radius) = m_lbp_xt->operator()(kxt, max_radius, max_radius);

            /*Getting the "micro-plane" for YT calculus*/
            const blitz::Array<T,2> kyt = 
               src(blitz::Range(i-max_radius,i+max_radius),blitz::Range(j-max_radius,j+max_radius),k);
            yt(i-max_radius,j-max_radius,k-max_radius) = m_lbp_yt->operator()(kyt, max_radius, max_radius);
          }
        }
      }
    }
} }

#endif /* BOB_IP_LBPTOP_H */
