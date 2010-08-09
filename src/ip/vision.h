#ifndef VISION_INC
#define VISION_INC

#include "core/general.h"

namespace Torch {

/** RGB pixel

    \begin{verbatim}
    r is the red component
    g is the green component
    b is the blue component
    \end{verbatim}

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

    \begin{verbatim}
    y is the Y component
    u is the U component
    v is the V component
    \end{verbatim}

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

    \begin{verbatim}
    x is the x coordinate
    y is the y coordinate
    \end{verbatim}

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

    \begin{verbatim}
    rho and theta determine the polar coordinates of a 2D point
    \end{verbatim}

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

    \begin{verbatim}
    x is the upper-left x coordinate
    y is the upper-left y coordinate
    w is the width
    h is the height
    \end{verbatim}

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

    \begin{verbatim}
    w is the width
    h is the height
    \end{verbatim}

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

    \begin{verbatim}
    tl is the top-left polar coordinate
    tr is the top-right polar coordinate
    bl is the bottom-left polar coordinate
    br is the bottom-right polar coordinate
    \end{verbatim}

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

    \begin{verbatim}
    n is the number of points
    p is the array of points
    \end{verbatim}

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

    \begin{verbatim}
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

    \end{verbatim}

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

    \begin{verbatim}
    r is the real part
    i is the imaginary part
    \end{verbatim}

    @author Sebastien Marcel (marcel@idiap.ch)
    @version 2.0
    \date
    @since 2.0
*/
struct sComplex
{
	double r, i;
};

}

#endif
