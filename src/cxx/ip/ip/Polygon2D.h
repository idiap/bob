/**
 * @file cxx/ip/ip/Polygon2D.h
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
#ifndef POLYGON2D_INC
#define POLYGON2D_INC

#include "ip/Point2D.h"

namespace bob {

	/** This class is designed to handle a polygon in 2D

	    @author Sebastien Marcel (marcel@idiap.ch)
	    @version 2.0
	    \date
	    @since 1.0
	*/
	class Polygon2D : public geomObject
	{
		bool is_allocated;

	public:
		/// number of vertex points
		int n;

		/// array of points with V[n]=V[0], V[n+1]=V[1]
		Point2D *V;

		//-----

		/** @name constructors */
		//@{
		///
		Polygon2D() {};

		///
		Polygon2D(int n_);

		///
		Polygon2D(int n_, Point2D *V_);
		//@}

		/// destructor
		virtual ~Polygon2D();

		//-----

		/// draw the polygon on an image
		virtual void draw(Image *image_, Color color_);

		//---

		/** test the inclusion of a point into the polygon

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
