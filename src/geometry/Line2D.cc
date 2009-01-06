#include "Line2D.h"
#include "Color.h"

namespace Torch {

void Line2D::saveFile(File *file)
{
   	P0.saveFile(file);
   	P1.saveFile(file);
}

void Line2D::loadFile(File *file)
{
   	P0.loadFile(file);
   	P1.loadFile(file);
}

const char *Line2D::sprint()
{
	sprintf(buf_sprint, "((%g, %g) (%g, %g))", P0.x, P0.y, P1.x, P1.y);

	return buf_sprint;
}

void Line2D::draw(Image *image, Color color)
{
   	// Warning: this is not exactly correct as a line is infinite
	// TODO: computes intersections of the line with the image and draw it

	image->drawLine((int)P0.x, (int)P0.y, (int)P1.x, (int)P1.y, color);
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
