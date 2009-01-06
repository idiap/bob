#ifndef SEGMENT2D_INC
#define SEGMENT2D_INC

#include "Point2D.h"

namespace Torch {

/** This class is designed to handle a segment in 2D

    A Segment2D is finite, starts at P0 and ends at P1.

    @author Sebastien Marcel (marcel@idiap.ch)
    @version 2.0
    \Date
    @since 1.0
*/
class Segment2D : public geomObject
{
public:
	/// first point supporting the segment
	Point2D P0;

	/// second point supporting the segment
	Point2D P1;

	//-----

	/** @name constructors */
	//@{
	///
	Segment2D() {};

	///
	Segment2D(Point2D P0_, Point2D P1_) { P0 = P0_; P1 = P1_; }
	//@}

	/// destructor
	virtual ~Segment2D() {};

	//-----

	/// save the segment
	virtual void saveFile(File *file);

	/// load a segment
	virtual void loadFile(File *file);

	/// return the segment into a string
	virtual const char *sprint();

	//-----

	/// draw the segment on an image
	virtual void draw(Image *image_, Color color_);
};

}

#endif
