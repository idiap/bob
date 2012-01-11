/**
 * @file cxx/ip/ip/Ray2D.h
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
#ifndef RAY2D_INC
#define RAY2D_INC

#include "ip/Point2D.h"

namespace bob {

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
