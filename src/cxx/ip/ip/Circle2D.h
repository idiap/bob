/**
 * @file cxx/ip/ip/Circle2D.h
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
#ifndef CIRCLE2D_INC
#define CIRCLE2D_INC

#include "ip/Point2D.h"

namespace bob {

	/** This class is designed to handle a 2D circle

	    @author Sebastien Marcel (marcel@idiap.ch)
	    @version 2.0
	    \date
	    @since 1.0
	*/
	class Circle2D : public geomObject
	{
	public:
		/// number of vertex points
		int n;

		/// array of points with V[n]=V[0], V[n+1]=V[1]
		Point2D *V;

		/// center of the circle
		Point2D center;

		/// radius of the circle
		double radius;

		//---

		/** @name constructors */
		//@{
		/// empty constructor
		Circle2D();

		/// creates a circle from 1 point given as 2 integers and a radius
		Circle2D(int xc_, int yc_, double radius_);

		/// creates a circle from 1 point and a radius
		Circle2D(const sPoint2D& center_, double radius_);
		//@}

		/// destructor
		virtual ~Circle2D();

		//-----

		/// reset a circle from 1 point given as 2 integers and a radius
		void reset(int xc_, int yc_, double radius_);

		/// reset a circle from 1 point and a radius
		void reset(const sPoint2D& center_, double radius_);

		//-----

		/// draw the circle in an image
		virtual void draw(Image *image_, Color color_);

	#ifdef HAVE_X11
		/** draw the object on a X11 display

		    #line_style_# can be LineSolid, LineOnOffDash or LineDoubleDash
		    #fill_style_# can be FillSolid, FillTiled, FillStippled or FillOpaqueStippled
		*/
		virtual void xdraw(Display *pDisplay_, Pixmap pixmap_, GC gc_, unsigned long color_, int line_width_ = 1, int line_style_ = LineSolid, bool fill_ = false, int fill_style_ = FillSolid);
	#endif

		//-----

		/** test the inclusion of a point into the circle

		    This method uses the winding number test.
		    You can find the original algorithm \URL[here]{http://geometryalgorithms.com/Archive/algorithm_0103/algorithm_0103.htm}.

		    @param P_ is a point,
		    @return the winding number (=0 only if #P_# is outside #V#)

		    @author Dan Sunday (http://www.softsurfer.com/)
		*/
		int wn_PnPoly(const Point2D& P_);

	};

}

#endif
