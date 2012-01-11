/**
 * @file cxx/ip/ip/Affine2D.h
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
#ifndef AFFINE2D_INC
#define AFFINE2D_INC

#include "ip/Matrix2D.h"
#include "ip/Vector2D.h"
#include "ip/Rectangle2D.h"

namespace bob {

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
