#ifndef RECTANGLE2D_INC
#define RECTANGLE2D_INC

#include "Point2D.h"
#include "Vector2D.h"

namespace Torch {

class Matrix2D;

/** This class is designed to handle a rectangle in 2D

    \begin{verbatim}

              P0            P1
	        +----------+
		|          |
		|          |
		|          |
		|          |
		|          |
		|          |
	        +----------+
              P3            P2

    \end{verbatim}

    @author Sebastien Marcel (marcel@idiap.ch)
    @version 2.1
    \Date
    @since 1.0
*/
class Rectangle2D : public geomObject
{
public:
	/** @name 2D points defining the rectangle */
	//@{
	///
	Point2D P0;

	///
	Point2D P1;

	///
	Point2D P2;

	///
	Point2D P3;
	//@}

	//----

	/** @name constructors */
	//@{
	///
	Rectangle2D() {};

	///
	Rectangle2D(Point2D P0_, Point2D P1_, Point2D P2_, Point2D P3_) { P0 = P0_; P1 = P1_; P2 = P2_; P3 = P3_; }

	///
	Rectangle2D(Point2D P0_, int w_, int h_);

	///
	Rectangle2D(int x_, int y_, int w_, int h_);

	///
	Rectangle2D(sRect2D r);
	//@}

	/// destructor
	virtual ~Rectangle2D() {};

	//-----

	/** @name reset functions */
	//@{
	///
	void reset(Point2D P0_, Point2D P1_, Point2D P2_, Point2D P3_);

	///
	void reset(Point2D P0_, int w_, int h_);

	///
	void reset(int x_, int y_, int w_, int h_);

	///
	void reset(sRect2D r_);
	//@}

	//-----

	/// save the rectangle
	virtual void saveFile(File *file);

	/// load a rectangle
	virtual void loadFile(File *file);

	/// return a rectangle into a string
	virtual const char *sprint();

	//-----

	/// draw the rectangle in an image
	virtual void draw(Image *image_, Color color_);

#ifdef HAVE_X11
	/** draw the object on a X11 display

            #line_style_# can be LineSolid, LineOnOffDash or LineDoubleDash
	    #fill_style_# can be FillSolid, FillTiled, FillStippled or FillOpaqueStippled
	*/
	virtual void xdraw(Display *pDisplay_, Pixmap pixmap_, GC gc_, unsigned long color_, int line_width_ = 1, int line_style_ = LineSolid, bool fill_ = false, int fill_style_ = FillSolid);
#endif

	//---

	/** @name Vector2D operations

	    These operations are not always valid. */
	//@{
	/// Vector translate +
	Rectangle2D  operator+(Vector2D);

	/// Vector translate -
	Rectangle2D  operator-(Vector2D);
	//@}

	///
	void fixI();
};

//-----

/** @name Rectangle2D rotation/scale

    @author Sebastien Marcel (marcel@idiap.ch)
    @version 2.1
    \Date
    @since 1.0
*/
//@{
/// Rectangle2D operator*(Rectangle2D, Matrix2D)
Rectangle2D operator*(Rectangle2D, Matrix2D);
//@}

//-----

/// returns true if the point is inside the rectangle
bool isInRect(int x_, int y_, sRect2D r_);

}

#endif
