/**
 * @file cxx/ip/ip/Rectangle2D.h
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
#ifndef RECTANGLE2D_INC
#define RECTANGLE2D_INC

#include "ip/Point2D.h"
#include "ip/Vector2D.h"

namespace bob {

	class Matrix2D;

	/** This class is designed to handle a rectangle in 2D

	    \verbatim

		      P0            P1
			+----------+
			|          |
			|          |
			|          |
			|          |
			|          |
			|          |
			+----------+
		      P3            P2

	    \endverbatim

	    @author Sebastien Marcel (marcel@idiap.ch)
	    @version 2.1
	    \date
	    @since 1.0
	*/
	class Rectangle2D : public geomObject
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

		///
		Point2D P3;
		//@}

		//----

		/** @name constructors */
		//@{
		///
		Rectangle2D() {};

		///
		Rectangle2D(const Point2D& P0_, const Point2D& P1_, const Point2D& P2_, const Point2D& P3_)
		{ P0 = P0_; P1 = P1_; P2 = P2_; P3 = P3_; }

		///
		Rectangle2D(const Point2D& P0_, int w_, int h_);

		///
		Rectangle2D(int x_, int y_, int w_, int h_);

		///
		Rectangle2D(const sRect2D& r);
		//@}

		/// destructor
		virtual ~Rectangle2D() {};

		//-----

		/** @name reset functions */
		//@{
		///
		void reset(const Point2D& P0_, const Point2D& P1_, const Point2D& P2_, const Point2D& P3_);

		///
		void reset(const Point2D& P0_, int w_, int h_);

		///
		void reset(int x_, int y_, int w_, int h_);

		///
		void reset(const sRect2D& r_);
		//@}


		//-----

		/// draw the rectangle in an image
		virtual void draw(Image *image_, Color color_);

	#ifdef HAVE_X11
		/** draw the object on a X11 display

		    #line_style_# can be LineSolid, LineOnOffDash or LineDoubleDash
		    #fill_style_# can be FillSolid, FillTiled, FillStippled or FillOpaqueStippled
		*/
		virtual void xdraw(Display *pDisplay_, Pixmap pixmap_, GC gc_, unsigned long color_, int line_width_ = 1, int line_style_ = LineSolid, bool fill_ = false, int fill_style_ = FillSolid);
	#endif

		//---

		/** @name Vector2D operations

		    These operations are not always valid. */
		//@{
		/// Vector translate +
		Rectangle2D  operator+(const Vector2D&);

		/// Vector translate -
		Rectangle2D  operator-(const Vector2D&);
		//@}

		///
		void fixI();
	};

	//-----

	/** @name Rectangle2D rotation/scale

	    @author Sebastien Marcel (marcel@idiap.ch)
	    @version 2.1
	    \date
	    @since 1.0
	*/
	//@{
	/// Rectangle2D operator*(Rectangle2D, Matrix2D)
	Rectangle2D operator*(const Rectangle2D&, const Matrix2D&);
	//@}

	//-----

	/// returns true if the point is inside the rectangle
	bool isInRect(int x_, int y_, const sRect2D& r_);

}

#endif
