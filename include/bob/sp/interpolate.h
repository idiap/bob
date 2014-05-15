/**
 * @date Mon Apr 25 18:06:14 2011 +0200
 * @author Laurent El Shafey <Laurent.El-Shafey@idiap.ch>
 *
 * @brief Implements various interpolation techniques for 1D and 2D blitz
 * arrays.
 *
 * Copyright (C) Idiap Research Institute, Martigny, Switzerland
 */

#ifndef BOB_SP_INTERPOLATE_H
#define BOB_SP_INTERPOLATE_H

#include <blitz/array.h>

namespace bob { namespace sp { namespace detail {
  /**
   * @brief Perform bilinear interpolation in the 1D src array for the
   *   given floating point coordinates.
   *
   * @param src The input blitz array.
   * @param x coordinate of the point to interpolate
   */
  template<typename T>
  double bilinearInterpolationNoCheck(const blitz::Array<T,1>& src,
      const double x)
  {
    int xl = static_cast<int>(floor(x));
    int xh = static_cast<int>(ceil(x));

    return (xh-x)*src(xl) + (1.-(xh-x))*src(xh);
  }

  /**
   * @brief Perform bilinear interpolation in the 1D src array for the
   *   given floating point coordinate, using image wrapping when coordinates
   *   are outside the image boundaries.
   *
   * @param src The input blitz array.
   * @param x coordinate of the point to interpolate
   */
  template<typename T>
  double bilinearInterpolationWrapNoCheck(const blitz::Array<T,1>& src,
      const double x)
  {
    const int res = src.extent()[0];
    // Wrap around values (the + res needs to be done since -1 % res == -1)
    int xl = (static_cast<int>(floor(x)) + res) % res;
    int xh = (static_cast<int>(ceil(x)) + res) % res;

    return (xh-x)*src(xl) + (1.-(xh-x))*src(xh);
  }


  /**
   * @brief Perform bilinear interpolation in the 2D src array for the
   *   given floating point coordinates.
   *
   * @param src The input blitz array.
   * @param y y-coordinate of the point to interpolate
   * @param x x-coordinate of the point to interpolate
   */
  template<typename T>
  double bilinearInterpolationNoCheck(const blitz::Array<T,2>& src,
      const double y, const double x)
  {
    int yl = static_cast<int>(floor(y));
    int yh = static_cast<int>(ceil(y));
    int xl = static_cast<int>(floor(x));
    int xh = static_cast<int>(ceil(x));

    const double Il = (xh-x)*src(yl,xl) + (1-(xh-x))*src(yl,xh);
    const double Ih = (xh-x)*src(yh,xl) + (1-(xh-x))*src(yh,xh);

    return (yh-y)*Il + (1.-(yh-y))*Ih;
  }

  /**
   * @brief Perform bilinear interpolation in the 2D src array for the
   *   given floating point coordinates, using image wrapping when coordinates
   *   are outside the image boundaries.
   *
   * @param src The input blitz array.
   * @param y y-coordinate of the point to interpolate
   * @param x x-coordinate of the point to interpolate
   */
  template<typename T>
  double bilinearInterpolationWrapNoCheck(const blitz::Array<T,2>& src,
      const double y, const double x)
  {
    const blitz::TinyVector<int,2>& res = src.extent();
    // Wrap around values (the + res needs to be done since -1 % res == -1)
    int yl = (static_cast<int>(floor(y)) + res[0]) % res[0];
    int yh = (static_cast<int>(ceil(y)) + res[0]) % res[0];
    int xl = (static_cast<int>(floor(x)) + res[1]) % res[1];
    int xh = (static_cast<int>(ceil(x)) + res[1]) % res[1];

    const double Il = (xh-x)*src(yl,xl) + (1-(xh-x))*src(yl,xh);
    const double Ih = (xh-x)*src(yh,xl) + (1-(xh-x))*src(yh,xh);

    return (yh-y)*Il + (1.-(yh-y))*Ih;
  }


}}}

#endif /* BOB_SP_INTERPOLATE_H */
