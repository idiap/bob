#ifndef RAY2D_INC
#define RAY2D_INC

#include "ip/Point2D.h"

namespace Torch {

	/** This class is designed to handle a ray in 2D

	    A Ray2D is semi-finite, starts at P0 and extends beyond P1

	    @author Sebastien Marcel (marcel@idiap.ch)
	    @version 2.0
	    \date
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
		Ray2D(const Point2D& P0_, const Point2D& P1_) { P0 = P0_; P1 = P1_; }
		//@}

		/// destructor
		virtual ~Ray2D() {};

		//-----

		/// draw the ray in an image
		virtual void draw(Image *image_, Color color_);
	};

}

#endif
