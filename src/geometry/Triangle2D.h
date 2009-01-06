#ifndef TRIANGLE2D_INC
#define TRIANGLE2D_INC

#include "Point2D.h"

namespace Torch {

/** This class is designed to handle a triangle in 2D

    @author Sebastien Marcel (marcel@idiap.ch)
    @version 2.0
    \Date
    @since 1.0
*/
class Triangle2D : public geomObject
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
	//@}

	//----

	/** @name constructors */
	//@{
	///
	Triangle2D() {};

	///
	Triangle2D(Point2D P0_, Point2D P1_, Point2D P2_) { P0 = P0_; P1 = P1_; P2 = P2_; };
	//@}

	/// destructor
	virtual ~Triangle2D() {};

	//-----

	/// save the triangle
	virtual void saveFile(File *file);

	/// load a triangle
	virtual void loadFile(File *file);

	/// return the triangle into a string
	virtual const char *sprint();

	//-----

	/// draw the triangle in an image
	virtual void draw(Image *image_, Color color_);

	//-----

	/// compute the angle (in radian) of 2 lines (P0,P1) and (P0,P2)
	double angle() { return Torch::angle(P0, P1, P2); };

	/// test the orientation of a triangle (angle direction of vectors (P0,P1) and (P0,P2))
	int ccw() { return Torch::ccw(P0, P1, P2); };
};

}

#endif
