#ifndef RAY2D_INC
#define RAY2D_INC

#include "Point2D.h"

namespace Torch {

/** This class is designed to handle a ray in 2D

    A Ray2D is semi-finite, starts at P0 and extends beyond P1

    @author Sebastien Marcel (marcel@idiap.ch)
    @version 2.0
    \Date
    @since 1.0
*/
class Ray2D : public geomObject
{
public:
	/// first point supporting the ray
	Point2D P0;

	/// second point supporting the ray
	Point2D P1;

	//-----

	/** @name constructors */
	//@{
	///
	Ray2D() {};

	///
	Ray2D(Point2D P0_, Point2D P1_) { P0 = P0_; P1 = P1_; }
	//@}

	/// destructor
	virtual ~Ray2D() {};

	//-----

	/// save the ray
	virtual void saveFile(File *file);

	/// load a ray
	virtual void loadFile(File *file);

	/// return the ray into a string
	virtual const char *sprint();

	//-----

	/// draw the ray in an image
	virtual void draw(Image *image_, Color color_);
};

}

#endif
