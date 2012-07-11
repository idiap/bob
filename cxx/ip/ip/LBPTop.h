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
       * @param m_lbp_xy The 2D LBP-XY plane configuration
       * @param m_lbp_xt The 2D LBP-XT plane configuration
       * @param m_lbp_yt The 2D LBP-YT plane configuration
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
          blitz::Array<uint16_t,2>& xy,
          blitz::Array<uint16_t,2>& xt,
          blitz::Array<uint16_t,2>& yt) const;

      void operator()(const blitz::Array<uint16_t,3>& src, 
          blitz::Array<uint16_t,2>& xy,
          blitz::Array<uint16_t,2>& xt,
          blitz::Array<uint16_t,2>& yt) const;

      void operator()(const blitz::Array<double,3>& src, 
          blitz::Array<uint16_t,2>& xy,
          blitz::Array<uint16_t,2>& xt,
          blitz::Array<uint16_t,2>& yt) const;

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
            blitz::Array<uint16_t,2>& xy,
            blitz::Array<uint16_t,2>& xt,
            blitz::Array<uint16_t,2>& yt) const;

      boost::shared_ptr<bob::ip::LBP> m_lbp_xy; ///< LBP for the XY calculation
      boost::shared_ptr<bob::ip::LBP> m_lbp_xt; ///< LBP for the XT calculation
      boost::shared_ptr<bob::ip::LBP> m_lbp_yt; ///< LBP for the YT calculation
  };

  /**
   * Implementation of certain template methods.
   */

  template <typename T>
    void bob::ip::LBPTop::process(const blitz::Array<T,3>& src,
                                  blitz::Array<uint16_t,2>& xy,
                                  blitz::Array<uint16_t,2>& xt,
                                  blitz::Array<uint16_t,2>& yt) const
    {

      int radius_xy = m_lbp_xy->getRadius(); ///< The LBPu2,i radius in XY
      int radius_xt = m_lbp_xt->getRadius(); ///< The LBPu2,i radius in XT
      int radius_yt = m_lbp_yt->getRadius(); ///< The LBPu2,i radius in YT

      int Tlength = src.extent(0);
      int height = src.extent(1);
      int width = src.extent(2);
      int tc = Tlength/2;
      int yc = height/2;
      int xc = width/2;

      int x=0,y=0,t=0;
      int correctX=0,correctY=0,correctT=0;

      /**** Get XY plane ****/
      const blitz::Array<T,2> kxy = 
        src( tc, blitz::Range::all(), blitz::Range::all());

      /*Checking the LBP conditions for XY. Just touching the method in order to stress theirs exceptions*/
      y=radius_xy; x=y;
      m_lbp_xy->operator()(kxy, y, x);

      /**** Get XT plane ****/
      const blitz::Array<T,2> kxt = 
        src( blitz::Range::all(), yc, blitz::Range::all());

      /*Checking the LBP conditions for XT. Just touching the method in order to stress theirs exceptions*/
      t=radius_xt; x = t;
      m_lbp_xt->operator()(kxt, t, x);

      /**** Get YT plane ****/
      const blitz::Array<T,2> kyt = 
        src( blitz::Range::all(), blitz::Range::all(), xc);

      /*Checking the LBP conditions for YT. Just touching the method in order to stress theirs exceptions*/
      t=radius_yt; y = t;
      m_lbp_yt->operator()(kyt, t, y);

      /**** Checking the size output arrays ****/
      /*Checking XY*/
      correctX = width-(radius_xy*2);
      correctY = height-(radius_xy*2);
      x = xy.extent(0);
      y = xy.extent(1);

      if(x > correctX)
        throw ParamOutOfBoundaryError("XY_Plane(x,y) x dimension",true,x,correctX);
      else if(x < correctX)
        throw ParamOutOfBoundaryError("XY_Plane(x,y) x dimension",false,x,correctX);

      if(y > correctY)
        throw ParamOutOfBoundaryError("XY_Plane(x,y) y dimension",true,y,correctY);
      else if(x < correctX)
        throw ParamOutOfBoundaryError("XY_Plane(x,y) y dimension",false,y,correctY);

      /*Checking XT*/
      correctT = Tlength-(radius_xt*2);
      correctX = width-(radius_xt*2);
      t = xt.extent(0);
      x = xt.extent(1);

      if(t > correctT)
        throw ParamOutOfBoundaryError("XT_Plane(t,x) t dimension",true,t,correctT);
      else if(t < correctT)
        throw ParamOutOfBoundaryError("XT_Plane(t,x) t dimension",false,t,correctT);

      if(x > correctX)
        throw ParamOutOfBoundaryError("XT_Plane(t,x) x dimension",true,x,correctX);
      else if(x < correctX)
        throw ParamOutOfBoundaryError("XT_Plane(t,x) x dimension",false,x,correctX);

      /*Checking YT*/
      correctY = height-(radius_yt*2);
      correctT = Tlength-(radius_yt*2);
      t = yt.extent(0);
      y = yt.extent(1);

      if(t > correctT)
        throw ParamOutOfBoundaryError("YT_Plane(t,y) t dimension",true,t,correctT);
      else if(t < correctT)
        throw ParamOutOfBoundaryError("YT_Plane(t,y) t dimension",false,t,correctT);

      if(y > correctY)
        throw ParamOutOfBoundaryError("YT_Plane(t,y) y dimension",true,y,correctY);
      else if(y < correctY)
        throw ParamOutOfBoundaryError("YT_Plane(t,y) y dimension",false,y,correctY);

      /*Calculating the XY plane*/
      for (int y=radius_xy; y < (height-radius_xy); ++y) {
        for (int x=radius_xy; x < (width-radius_xy); ++x) {
          xy(y-radius_xy,x-radius_xy) = m_lbp_xy->operator()(kxy, y, x);
        }
      }

      /*Calculation of the XT plane*/
      for (int t = radius_xt; t < (Tlength-radius_xt); ++t) {
        for (int x=radius_xt; x < (width-radius_xt); ++x) {
          xt(t-radius_xt,x-radius_xt) = m_lbp_xt->operator()(kxt, t, x);
        }
      }

      /*Calculation of the YT plane*/
      for (int t = radius_yt; t < (Tlength-radius_yt); ++t) {
        for (int y = radius_yt; y < (height-radius_yt); ++y) {
          yt(t-radius_yt,y-radius_yt) = m_lbp_yt->operator()(kyt, t, y);
        }
      }

    }

} }

#endif /* BOB_IP_LBPTOP_H */
