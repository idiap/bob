/**
 * @file cxx/ip/ip/vision.h
 * @date Tue Jan 18 17:07:26 2011 +0100
 * @author André Anjos <andre.anjos@idiap.ch>
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
#ifndef VISION_INC
#define VISION_INC

#include "core/general.h"

namespace Torch {

/** RGB pixel

    \verbatim
    r is the red component
    g is the green component
    b is the blue component
    \endverbatim

    @author Sebastien Marcel (marcel@idiap.ch)
    @version 2.0
    \date
    @since 1.0
*/
struct sPixRGB
{
	unsigned char r, g, b;
};

/** YUV pixel

    \verbatim
    y is the Y component
    u is the U component
    v is the V component
    \endverbatim

    @author Sebastien Marcel (marcel@idiap.ch)
    @version 2.0
    \date
    @since 1.0
*/
struct sPixYUV
{
	unsigned char y, u, v;
};

/** Point2D (in double)

    \verbatim
    x is the x coordinate
    y is the y coordinate
    \endverbatim

    @author Sebastien Marcel (marcel@idiap.ch)
    @version 2.0
    \date
    @since 1.0
*/
struct sPoint2D
{
	sPoint2D(double x_ = 0, double y_ = 0) : x(x_), y(y_)
	{
	}

	double x, y;
};

/** Point2D (in polar coordinates)

    \verbatim
    rho and theta determine the polar coordinates of a 2D point
    \endverbatim

    @author Sebastien Marcel (marcel@idiap.ch)
    @version 2.1
    \date
    @since 2.1
*/
struct sPoint2Dpolar
{
	double rho, theta;
};

/** Rect2D

    \verbatim
    x is the upper-left x coordinate
    y is the upper-left y coordinate
    w is the width
    h is the height
    \endverbatim

    @author Sebastien Marcel (marcel@idiap.ch)
    @version 2.0
    \date
    @since 2.0
*/
struct sRect2D
{
	// Constructor
	sRect2D(int x_ = 0, int y_ = 0, int w_ = 0, int h_ = 0)
		: x(x_), y(y_), w(w_), h(h_)
	{
	}

	// Attributes
	int x, y, w, h;
};

/** sSize

    \verbatim
    w is the width
    h is the height
    \endverbatim

    @author Sebastien Marcel (marcel@idiap.ch)
    @version 2.0
    \date
    @since 2.0
*/
struct sSize
{
	// Constructor
	sSize(int new_w = 0, int new_h = 0)
		:	w(new_w), h(new_h)
	{
	}

	// Attributes
	int		w;
	int		h;
};


/** Rect2D (in polar coordinates)

    \verbatim
    tl is the top-left polar coordinate
    tr is the top-right polar coordinate
    bl is the bottom-left polar coordinate
    br is the bottom-right polar coordinate
    \endverbatim

    @author Sebastien Marcel (marcel@idiap.ch)
    @version 2.1
    \date
    @since 2.1
*/
struct sRect2Dpolar
{
	sPoint2Dpolar tl, tr, bl, br;
};

/** Poly2D

    \verbatim
    n is the number of points
    p is the array of points
    \endverbatim

    @author Sebastien Marcel (marcel@idiap.ch)
    @version 2.0
    \date
    @since 2.0
*/
struct sPoly2D
{
	int n;
	sPoint2D *p;
};

/** Octon2D

    An octon is a particular octogon with horizontal top/bottom and vertical left/right.

    \verbatim
              __________
             /          \
            /            |
           |             |
           |             |
           |             |
           |            /
            \          /
             \________/

	surface is the number of pixels
	width is the width
	height is the height
	cg is the center of gravity

    \endverbatim

    @author Sebastien Marcel (marcel@idiap.ch)
    @version 2.0
    \date
    @since 2.0
*/
struct sOcton
{
	int surface;
	int height;
	int width;
	sPoint2D cg;
        int y_min, y_max;
        int x_min, x_max;
        int ypx_min, ypx_max, ymx_min, ymx_max;
};

/** Complex (in double)

    \verbatim
    r is the real part
    i is the imaginary part
    \endverbatim

    @author Sebastien Marcel (marcel@idiap.ch)
    @version 2.0
    \date
    @since 2.0
*/
struct sComplex
{
	double r, i;
};

// Global object to "inform" ipXXX classes that they are used for multiscale,
// pyramid	or some other scanning procedure.
// WARNING: This code was moved from "scanning" to avoid the circular
// dependence between "ip" and "scanning". REVISE THIS AS SOON AS POSSIBLE.
enum ScanType
{
  ScanTypeMultiscale,
  ScanTypePyramid,
  ScanTypeOther
};
class CurrentScanType
{
public:

  static CurrentScanType& getInstance()
  {
    static CurrentScanType instance;
    return instance;
  }

  ~CurrentScanType() {}

  ScanType	get() const { return m_value; }
  void		set(ScanType value) { m_value = value; }

private:

  CurrentScanType() : m_value(ScanTypePyramid) {}
  CurrentScanType(const CurrentScanType& other);
  CurrentScanType& operator=(const CurrentScanType& other);

  ScanType	m_value;
};

}

#endif
