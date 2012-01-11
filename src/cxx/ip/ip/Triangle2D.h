/**
 * @file cxx/ip/ip/Triangle2D.h
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
#ifndef TRIANGLE2D_INC
#define TRIANGLE2D_INC

#include "ip/Point2D.h"

namespace bob {

	/** This class is designed to handle a triangle in 2D

	    @author Sebastien Marcel (marcel@idiap.ch)
	    @version 2.0
	    \date
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
		Triangle2D(const Point2D& P0_, const Point2D& P1_, const Point2D& P2_)
		{ P0 = P0_; P1 = P1_; P2 = P2_; };
		//@}

		/// destructor
		virtual ~Triangle2D() {};

		//-----

		/// draw the triangle in an image
		virtual void draw(Image *image_, Color color_);

		//-----

		/// compute the angle (in radian) of 2 lines (P0,P1) and (P0,P2)
		double angle() { return bob::angle(P0, P1, P2); };

		/// test the orientation of a triangle (angle direction of vectors (P0,P1) and (P0,P2))
		int ccw() { return bob::ccw(P0, P1, P2); };
	};

}

#endif
