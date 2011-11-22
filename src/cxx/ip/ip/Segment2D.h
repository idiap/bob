/**
 * @file cxx/ip/ip/Segment2D.h
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
#ifndef SEGMENT2D_INC
#define SEGMENT2D_INC

#include "ip/Point2D.h"

namespace Torch {

	/** This class is designed to handle a segment in 2D

	    A Segment2D is finite, starts at P0 and ends at P1.

	    @author Sebastien Marcel (marcel@idiap.ch)
	    @version 2.0
	    \date
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
		Segment2D(const Point2D& P0_, const Point2D& P1_) { P0 = P0_; P1 = P1_; }
		//@}

		/// destructor
		virtual ~Segment2D() {};

		//-----

		/// draw the segment on an image
		virtual void draw(Image *image_, Color color_);
	};

}

#endif
