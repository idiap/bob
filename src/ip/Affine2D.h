#ifndef AFFINE2D_INC
#define AFFINE2D_INC

#include "Matrix2D.h"
#include "Vector2D.h"
#include "Rectangle2D.h"

namespace Torch {

	/** This class is designed to handle 2D affine (rotation, scale and translation) transformations

	    @author Sebastien Marcel (marcel@idiap.ch)
	    @version 2.0
	    \date
	    @since 1.0
	*/
	class Affine2D
	{
	public:
		/// rotation/scale matrix
		Matrix2D rs;

		/// translation vector
		Vector2D t;

		//-----

		/** constructor

		    @param rs_ is a rotation/scale matrix
		    @param r_ is a translation vector
		*/
		Affine2D(Matrix2D &rs_, Vector2D &t_);

		/// destructor
		~Affine2D() {};

		/** @name affine transformation (#v# - #t#) * #rs#
		*/
		//@{
		///
		Point2D operator*(Point2D &p);

		///
		Vector2D operator*(Vector2D &v);

		///
		Rectangle2D operator*(Rectangle2D &v);
		//@}
	};

}

#endif
