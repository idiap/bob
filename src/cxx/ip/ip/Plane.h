/**
 * @file cxx/ip/ip/Plane.h
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
#ifndef PLANE_INC
#define PLANE_INC

#include "ip/Point3D.h"
#include "ip/Vector3D.h"

namespace bob {

	/** This class is designed to handle a plane

	    A Plane is defined by a point P0 and a normal n.

	    @author Sebastien Marcel (marcel@idiap.ch)
	    @author Dan Sunday (http://www.softsurfer.com/)
	    @version 2.0
	    \date
	    @since 1.0
	*/
	class Plane : public geomObject
	{
	public:
		/// 3D point
		Point3D P0;

		/// 3D normal vector
		Vector3D n;

		//-----

		/** @name constructors */
		//@{
		/// empty constructor
		Plane() {};
		/// creates a plane from a point and a normal
		Plane(const Point3D& P0_, const Vector3D& n_) { P0 = P0_; n = n_; }
		//@}

		/// destructor
		virtual ~Plane() {};

		//-----

		/// draw the plane on a image
		virtual void draw(Image *image_, Color color_);
	};

}

#endif
