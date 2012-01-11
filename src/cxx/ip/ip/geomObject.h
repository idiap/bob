/**
 * @file cxx/ip/ip/geomObject.h
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
#ifndef GEOM_OBJECT_INC
#define GEOM_OBJECT_INC

#include "ip/Image.h"
#include "ip/vision.h"

#ifdef HAVE_X11
#include "X.h"
#endif

namespace bob {

	/** This class is designed to handle geometry objects and algorithms

	    @author Sebastien Marcel (marcel@idiap.ch)

	    @version 2.0
	    \date
	    @since 1.0
	*/
	class geomObject
	{
	public:
		/// verbose
		bool verbose;

		//-----

		/// constructor
		geomObject() { verbose = false; }

		/// destructor
		virtual ~geomObject() {};

		//-----

		/// draw the object on an image
		virtual void draw(Image *image_, Color color_) = 0;

	#ifdef HAVE_X11
		/** draw the object on a X11 display

		    #line_style_# can be LineSolid, LineOnOffDash or LineDoubleDash
		    #fill_style_# can be FillSolid, FillTiled, FillStippled or FillOpaqueStippled
		*/
		virtual void xdraw(Display *pDisplay_, Pixmap pixmap_, GC gc_, unsigned long color_, int line_width_ = 1, int line_style_ = LineSolid, bool fill_ = false, int fill_style_ = FillSolid) {};
	#endif
	};

}

#endif
