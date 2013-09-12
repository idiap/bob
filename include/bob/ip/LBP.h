/**
 * @file bob/ip/LBP.h
 * @date Wed Apr 20 20:21:19 2011 +0200
 * @author Laurent El Shafey <Laurent.El-Shafey@idiap.ch>
 *
 * Rewritten:
 * @date Wed Apr 10 17:39:21 CEST 2013
 * @author Manuel Günther <manuel.guenther@idiap.ch>
 *
 * This file defines a class to compute LBP and variants
 *
 * Copyright (C) 2011-2013 Idiap Research Institute, Martigny, Switzerland
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

#ifndef BOB_IP_LBP_H
#define BOB_IP_LBP_H

#include <math.h>
#include <stdint.h>
#include <numeric>
#include <stdexcept>
#include <boost/format.hpp>

#include <blitz/array.h>

#include <bob/core/assert.h>
#include <bob/sp/interpolate.h>
#include <bob/ip/integral.h>


namespace bob { namespace ip {

  /**
   * Different ways to extract LBP codes: regular, transitional or direction coded (see Cosmin's thesis)
   */
  typedef enum{
    ELBP_REGULAR = 0,       //!< regular LBP codes: each pixel value is compared to the center
    ELBP_TRANSITIONAL = 1,  //!< transitional LBP codes: instead of comparing to the center, the pixel value is compared to the next neighbor
    ELBP_DIRECTION_CODED = 2//!< direction coded LBP: each three pixel values in a row define a two bit codes, which are then connected
  } ELBPType;

  typedef enum{
    LBP_BORDER_SHRINK,      //!< shrink the resulting image by 2* radius
    LBP_BORDER_WRAP         //!< wrap around the image so that pixel[-1] == pixel[res - 1]
  } LBPBorderHandling;

  /**
   * This class is an abstraction for all the Local Binary Patterns
   *   variants. For more information, please refer to the following
   *   article:
   *     "Face Recognition with Local Binary Patterns", from T. Ahonen,
   *     A. Hadid and M. Pietikainen
   *     in the proceedings of the European Conference on Computer Vision
   *     (ECCV'2004), p. 469-481
   *
   *   All implemented variants are listed in the thesis of Cosmin Atanasoaei
   *   "Multivariate Boosting with Look-Up Tables for Face Processing"
   *   http://publications.idiap.ch/index.php/publications/show/2315
   *
   */
  class LBP {

    public: //api

      /**
       * Complete constructor with two radii. This will permit extraction of elliptical and rectangular LBP codes.
       *
       * @param P     number of neighbors
       * @param R_y   radius in vertical direction
       * @param R_x   radius in horizontal direction
       * @param circular  circular or rectangular LBP
       * @param to_average  compare to average value or to central pixel value
       * @param add_average_bit  if to_average: also add a bit for the comparison of the central bit with the average
       * @param uniform  compute LBP^u2 uniform LBP's (see paper listed above)
       * @param rotation_invariant  compute rotation invariant LBP's
       * @param eLBP_type  The extended type of LBP: regular, transitional or direction coded (see Cosmin's thesis)
       * @param border_handling  How to handle the image in border cases
       */
      LBP(const int P,
          const double R_y,
          const double R_x,
          const bool circular=false,
          const bool to_average=false,
          const bool add_average_bit=false,
          const bool uniform=false,
          const bool rotation_invariant=false,
          const bob::ip::ELBPType eLBP_type=ELBP_REGULAR,
          const bob::ip::LBPBorderHandling border_handling=LBP_BORDER_SHRINK);

      /**
       * Complete constructor with one radius. This will permit extraction of round and square LBP codes.
       * This constructor implements the default behavior of both papers mentioned above.
       *
       * @param P     number of neighbors
       * @param R     radius in both horizontal and vertical direction
       * @param circular  circular or rectangular LBP
       * @param to_average  compare to average value or to central pixel value
       * @param add_average_bit  if to_average: also add a bit for the comparison of the central bit with the average
       * @param uniform  compute LBP^u2 uniform LBP's (see paper listed above)
       * @param rotation_invariant  compute rotation invariant LBP's
       * @param eLBP_type  The extended type of LBP: REGULAR, TRANSITIONAL or DIRECTION_CODED (see Cosmins thesis)
       * @param border_handling  How to handle the image in border cases
       */
      LBP(const int P,
          const double R=1.,
          const bool circular=false,
          const bool to_average=false,
          const bool add_average_bit=false,
          const bool uniform=false,
          const bool rotation_invariant=false,
          const bob::ip::ELBPType eLBP_type=ELBP_REGULAR,
          const bob::ip::LBPBorderHandling border_handling=LBP_BORDER_SHRINK);


      /**
       * Constructor for multi-block LBP codes
       * @param P     number of neighbors
       * @param block_size   the block size for the multi-block LBP in both y and x direction
       * @param to_average  compare to average value or to central pixel value
       * @param add_average_bit  if to_average: also add a bit for the comparison of the central bit with the average
       * @param uniform  compute LBP^u2 uniform LBP's (see paper listed above)
       * @param rotation_invariant  compute rotation invariant LBP's
       * @param eLBP_type  The extended type of LBP: regular, transitional or direction coded (see Cosmin's thesis)
       * @param border_handling  How to handle the image in border cases
       */
      LBP(const int P,
          const blitz::TinyVector<int,2> block_size,
          const bool to_average=false,
          const bool add_average_bit=false,
          const bool uniform=false,
          const bool rotation_invariant=false,
          const bob::ip::ELBPType eLBP_type=ELBP_REGULAR,
          const bob::ip::LBPBorderHandling border_handling=LBP_BORDER_SHRINK);

      /**
       * Copy constructor
       */
      LBP(const LBP& other);

      /**
       * Destructor
       */
      virtual ~LBP();

      /**
       * Assignment
       */
      LBP& operator= (const bob::ip::LBP& other);

      /**
       * Return the maximum number of labels for the current LBP variant
       */
      int getMaxLabel() const;

      /**
       * Accessors
       */
      double getRadius() const {
        if (m_R_x != m_R_y) {
          boost::format m("the radii R_x (%f) and R_y (%f) do not match");
          m % m_R_x % m_R_y;
          throw std::runtime_error(m.str());
        }
        return m_R_x;
      }
      blitz::TinyVector<double,2> getRadii() const { return blitz::TinyVector<double,2>(m_R_y, m_R_x); }
      blitz::TinyVector<double,2> getBlockSize() const { return blitz::TinyVector<int,2>(m_mb_y, m_mb_x); }
      int getNNeighbours() const { return m_P; }
      bool getCircular() const { return m_circular; }
      bool getToAverage() const { return m_to_average; }
      bool getAddAverageBit() const { return m_add_average_bit; }
      bool getUniform() const { return m_uniform; }
      bool getRotationInvariant() const { return m_rotation_invariant; }
      bob::ip::ELBPType get_eLBP() const { return m_eLBP_type; }
      bob::ip::LBPBorderHandling getBorderHandling() const { return m_border_handling; }
      blitz::Array<double,2> getRelativePositions(){return m_positions;}
      blitz::Array<uint16_t,1> getLookUpTable(){return m_lut;}

      /**
       * Mutators
       */
      void setRadius(const double R){ m_R_y = R; m_R_x = R; init(); }
      void setRadii(blitz::TinyVector<double,2> r){ m_R_y = r[0]; m_R_x = r[1]; init(); }
      void setBlockSize(blitz::TinyVector<int,2> mb){ m_mb_y = mb[0]; m_mb_x = mb[1]; init(); }
      void setNNeighbours(const int neighbors) { m_P = neighbors; init(); }
      void setCircular(const bool circ){ m_circular = circ; init(); }
      void setToAverage(const bool to_average){ m_to_average = to_average; init(); }
      void setAddAverageBit(const bool add_average_bit){ m_add_average_bit = add_average_bit; init(); }
      void setUniform(const bool uniform){ m_uniform = uniform; init(); }
      void setRotationInvariant(const bool rotation_invariant){ m_rotation_invariant = rotation_invariant; init(); }
      void set_eLBP(bob::ip::ELBPType eLBP_type){ m_eLBP_type = eLBP_type; if (eLBP_type == ELBP_DIRECTION_CODED && m_P%2) { throw std::runtime_error("direction coded LBP types require an even number of neighbors.");}}
      void setBorderHandling(bob::ip::LBPBorderHandling border_handling){ m_border_handling = border_handling; }
      void setLookUpTable(const blitz::Array<uint16_t,1>& new_lut){m_lut = new_lut;}

      /**
       * Extract LBP features from a 2D blitz::Array, and save
       *   the resulting LBP codes in the dst 2D blitz::Array.
       *   For multi-block LBP types, the given image might be an integral image.
       *   Please set is_integral_image to true in this case
       */
      template <typename T>
        void operator()(const blitz::Array<T,2>& src, blitz::Array<uint16_t,2>& dst, bool is_integral_image = false) const;

      /**
       * Extract the LBP code of a 2D blitz::Array at the given
       *   location, and return it.
       *   For multi-block LBP types, the given image might be an integral image.
       *   Please set is_integral_image to true in this case.
       *   Assure that the integral image is one element larger than the original image (i.e., using the integral(src, ii, true) function).
       */
      template <typename T>
        uint16_t operator()(const blitz::Array<T,2>& src, int y, int x, bool is_integral_image = false) const;

      /**
       * Get the required shape of the dst output blitz array,
       *   before calling the operator() method.
       */
      template <typename T>
        const blitz::TinyVector<int,2> getLBPShape(const blitz::Array<T,2>& src, bool is_integral_image = false) const;

    private:

      /**
       * Initialize the look up table and the relative positions for the current setup
       */
      void init();

      /**
       * Circular shift to the right of the input pattern for shift positions
       */
      uint16_t right_shift_circular(uint16_t pattern, int shift);

      /**
       * Computes the LBP image from the given image.
       * For multi-block LBP features, the src image must be an integral image,
       * for other types of LBP it is not.
       */
      template <typename T>
        void apply(const blitz::Array<T,2>& src, blitz::Array<uint16_t,2>& dst) const;

      /**
       * Extract the LBP code of a 2D blitz::Array at the given location, and return it.
       * For multi-block LBP, the given image must be an integral image
       * Checks are disabled in this function.
       */
      template <typename T>
        uint16_t lbp_code(const blitz::Array<T,2>& src, int y, int x) const;


      /**
       * Attributes
       */
      int m_P;
      double m_R_y;
      double m_R_x;
      int m_mb_y; // size of multi-block block in y
      int m_mb_x; // size of multi-block block in x
      bool m_circular;
      bool m_to_average;
      bool m_add_average_bit;
      bool m_uniform;
      bool m_rotation_invariant;
      bob::ip::ELBPType m_eLBP_type;
      bob::ip::LBPBorderHandling m_border_handling;

      // the look up table for the current type of LBP (uniform, rotation-invariant, ...)
      blitz::Array<uint16_t,1> m_lut;

      // the positions of the points that have to be processed
      blitz::Array<double, 2> m_positions;

      // a pre-allocated copy of the integral image, just for speed purposes
      mutable blitz::Array<double, 2> _integral_image;
  };

  ///////////////////////////////////////////////////
  ///////// template function implementations ///////
  ///////////////////////////////////////////////////

  template <typename T>
    const blitz::TinyVector<int,2> LBP::getLBPShape(const blitz::Array<T,2>& src, bool is_integral_image) const
    {
      int dy, dx;
      if (m_border_handling == LBP_BORDER_WRAP){
        // when wrapping borders, the resolution is not altered
        dy = 0;
        dx = 0;
      } else if (m_mb_y > 0 && m_mb_x > 0){
        dy = 3 * m_mb_y - 1;
        dx = 3 * m_mb_x - 1;
      } else {
        dy = 2*(int)ceil(m_R_y);
        dx = 2*(int)ceil(m_R_x);
      }

      if (is_integral_image){
        // if the given image is an integral image, we have to subtract one pixel more
        dy += 1;
        dx += 1;
      }
      return blitz::TinyVector<int,2> (std::max(0, src.extent(0) - dy), std::max(0, src.extent(1) - dx));
    }


  template <typename T>
    inline void LBP::operator()(const blitz::Array<T,2>& src, blitz::Array<uint16_t,2>& dst, bool is_integral_image) const
    {
      bob::core::array::assertZeroBase(src);
      bob::core::array::assertZeroBase(dst);
      bob::core::array::assertSameShape(dst, getLBPShape(src, is_integral_image) );

      if (m_mb_y > 0 && m_mb_x > 0 &&! is_integral_image){
        // apply integral image
        _integral_image.resize(src.extent(0)+1, src.extent(1)+1);
        bob::ip::integral(src, _integral_image, true);
        apply<double>(_integral_image, dst);
      } else {
        apply<T>(src, dst);
      }
    }

    template <typename T>
      inline void LBP::apply(const blitz::Array<T,2>& src, blitz::Array<uint16_t,2>& dst) const
    {
      // offset in the source image
      int r_y, r_x;
      if (m_border_handling == LBP_BORDER_WRAP){
        r_y = 0;
        r_x = 0;
      } else if (m_mb_y > 0 && m_mb_x > 0){
        // + 1 since the integral image is one element larger than the original image
        r_y = m_mb_y + m_mb_y/2 + 1;
        r_x = m_mb_x + m_mb_x/2 + 1;
      } else {
        r_y = (int)ceil(m_R_y);
        r_x = (int)ceil(m_R_x);
      }

      // iterate over target pixels
      for (int y = 0; y < dst.extent(0); ++y)
        for (int x = 0; x < dst.extent(1); ++x)
          dst(y, x) = lbp_code(src, y + r_y, x + r_x);
    }

  template <typename T>
  inline uint16_t LBP::operator()(const blitz::Array<T,2>& src, int y, int x, bool is_integral_image) const{
    // perform some checks
    bob::core::array::assertZeroBase(src);
    // offset in the source image
    const int r_y = (int)ceil(m_R_y), r_x = (int)ceil(m_R_x);
    if (y < r_y || y >= src.extent(0)-r_y) {
     boost::format m("argument `y' = %d is set outside the expected range [%d, %d]");
     m % y % r_y % (src.extent(0)-r_y-1);
     throw std::runtime_error(m.str());
    }
    if (x < r_x || x >= src.extent(1)-r_x) {
     boost::format m("argument `x' = %d is set outside the expected range [%d, %d]");
     m % x % r_x % (src.extent(1)-r_x-1);
     throw std::runtime_error(m.str());
    }

    if (m_mb_y > 0 && m_mb_x > 0 && !is_integral_image){
      // apply integral image
      _integral_image.resize(src.extent(0)+1, src.extent(1)+1);
      // compute integral image; adds one line of zeros in the front
      bob::ip::integral(src, _integral_image, true);
      // return LBP code from integral image
      return lbp_code<double>(_integral_image, y, x);
    } else {
      // return LBP code from source image
      return lbp_code<T>(src, y, x);
    }
  }


  // implementation of the LBP code extraction
  template <typename T>
  inline uint16_t LBP::lbp_code(const blitz::Array<T,2>& src, int y, int x) const{
    std::vector<double> pixels(m_P);
    double center;
    if (m_mb_y > 0 && m_mb_x > 0){
      // extract the pixels from the INTEGRAL image
      // by wrapping around (also works for shrinking since these positions will never be used)
      for (int p = 0; p < m_P; ++p){
        const int y0 = (y + static_cast<int>(m_positions(p,0)) + src.extent(0)) % src.extent(0),
                  y1 = (y + static_cast<int>(m_positions(p,1)) + src.extent(0)) % src.extent(0),
                  x0 = (x + static_cast<int>(m_positions(p,2)) + src.extent(1)) % src.extent(1),
                  x1 = (x + static_cast<int>(m_positions(p,3)) + src.extent(1)) % src.extent(1);
        pixels[p] = static_cast<double>(src(y0, x0)) + static_cast<double>(src(y1, x1)) - static_cast<double>(src(y0, x1)) - static_cast<double>(src(y1, x0));
      }
      const int y0 = (y + static_cast<int>(m_positions(3,0)) + src.extent(0)) % src.extent(0),
                y1 = (y + static_cast<int>(m_positions(3,1)) + src.extent(0)) % src.extent(0),
                x0 = (x + static_cast<int>(m_positions(1,2)) + src.extent(1)) % src.extent(1),
                x1 = (x + static_cast<int>(m_positions(1,3)) + src.extent(1)) % src.extent(1);
      center = static_cast<double>(src(y0, x0)) + static_cast<double>(src(y1, x1)) - static_cast<double>(src(y0, x1)) - static_cast<double>(src(y1, x0));

    }else if (m_circular){
      // extract the pixels from the image by interpolating the image
      for (int p = 0; p < m_P; ++p)
        pixels[p] = bob::sp::detail::bilinearInterpolationWrapNoCheck(src, y + m_positions(p,0), x + m_positions(p,1));
      center = static_cast<double>(src(y, x));
    }else{
      // extract the pixels from the image by wrapping around (also works for shrinking since these positions will never be used)
      for (int p = 0; p < m_P; ++p){
        const int cy = (y + static_cast<int>(m_positions(p,0)) + src.extent(0)) % src.extent(0);
        const int cx = (x + static_cast<int>(m_positions(p,1)) + src.extent(1)) % src.extent(1);
        pixels[p] = static_cast<double>(src(cy, cx));
      }
      center = static_cast<double>(src(y, x));
    }


    double cmp_point = center;
    if (m_to_average)
      cmp_point = std::accumulate(pixels.begin(), pixels.end(), center) / (m_P + 1); // /(P+1) since (averaged over P+1 points)

    // the formulas are implemented from Cosmin's thesis
    uint16_t lbp_code = 0;
    switch (m_eLBP_type){
      case ELBP_REGULAR:{
        for (int p = 0; p < m_P; ++p){
          lbp_code <<= 1;
          if (pixels[p] > cmp_point || bob::core::isClose(pixels[p], cmp_point)) ++lbp_code;
        }
        if (m_add_average_bit && !m_rotation_invariant && !m_uniform)
        {
          lbp_code <<= 1;
          if (center > cmp_point || bob::core::isClose(center, cmp_point)) ++lbp_code;
        }
        break;
      }

      case ELBP_TRANSITIONAL:{
        for (int p = 0; p < m_P; ++p){
          lbp_code <<= 1;
          if (pixels[p] > pixels[(p+1)%m_P] || bob::core::isClose(pixels[p], pixels[(p+1)%m_P])) ++lbp_code;
        }
        break;
      }

      case ELBP_DIRECTION_CODED:{
        int p_half = m_P/2;
        for (int p = 0; p < p_half; ++p){
          lbp_code <<= 2;
          if ((pixels[p] - cmp_point) * (pixels[p+p_half] - cmp_point) >= 0.) lbp_code += 1;
          double p1 = std::abs(pixels[p] - cmp_point), p2 = std::abs(pixels[p+p_half] - cmp_point);
          if ( p1 > p2 || bob::core::isClose(p1, p2) ) lbp_code += 2;
        }
        break;
      }
    }

    // convert the lbp code according to the requested setup (uniform, rotation invariant, ...)
    return m_lut(lbp_code);
  }

} }

#endif /* BOB_IP_LBP_H */
