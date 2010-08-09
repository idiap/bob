#ifndef OCTON2D_INC
#define OCTON2D_INC

#include "Point2D.h"

namespace Torch {

	/** This class is designed to handle an octon in 2D

	    @author Sebastien Marcel (marcel@idiap.ch)
	    @version 2.0
	    \date
	    @since 2.0
	*/
	class Octon2D : public geomObject
	{
	public:
		sOcton octon;

		//-----

		/** @name constructors */
		//@{
		///
		Octon2D() {};

		///
		Octon2D(const sOcton& octon_) { octon = octon_; };
		//@}

		/// destructor
		virtual ~Octon2D();

		//-----

		/// draw the octon on an image
		virtual void draw(Image *image_, Color color_);

	#ifdef HAVE_X11
		/** draw the object on a X11 display

		    #line_style_# can be LineSolid, LineOnOffDash or LineDoubleDash
		    #fill_style_# can be FillSolid, FillTiled, FillStippled or FillOpaqueStippled
		*/
		virtual void xdraw(Display *pDisplay_, Pixmap pixmap_, GC gc_, unsigned long color_, int line_width_ = 1, int line_style_ = LineSolid, bool fill_ = false, int fill_style_ = FillSolid);
	#endif

		//---

		/** test the inclusion of a point into the octon

		    @param P_ is a point,
		    @return true if the point is included
		*/
		bool inside(const Point2D& P_);
	};

}

#endif
