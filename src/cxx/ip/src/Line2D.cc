/**
 * @file cxx/ip/src/Line2D.cc
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
#include "ip/Line2D.h"
#include "ip/OldColor.h"

namespace Torch {

void Line2D::draw(Image *image, Color color)
{
   	// Warning: this is not exactly correct as a line is infinite
	// TODO: computes intersections of the line with the image and draw it

	image->drawLine((int)P0.get(0), (int)P0.get(1), (int)P1.get(0), (int)P1.get(1), color);
}

#ifdef HAVE_X11
void Line2D::xdraw(Display *pDisplay, Pixmap pixmap, GC gc, unsigned long color, int line_width, int line_style, bool fill, int fill_style)
{
	XGCValues old_gc_values;
	XGCValues new_gc_values;

	int myChangeFlags = GCLineWidth|GCLineStyle;

	XSetForeground(pDisplay, gc, color);

	// Save current graphic context attributes
	XGetGCValues(pDisplay, gc, myChangeFlags, &old_gc_values);

	// Set new attributes
	new_gc_values.line_width = line_width;
	new_gc_values.line_style = line_style;

	XChangeGC(pDisplay, gc, myChangeFlags, &new_gc_values);

	XDrawLine(pDisplay, pixmap, gc, (int) P0.x, (int) P0.y, (int) P1.x, (int) P1.y);

	// Restore graphic context attributes
	XChangeGC(pDisplay, gc, myChangeFlags, &old_gc_values);
}
#endif

}
