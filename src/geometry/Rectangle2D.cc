#include "Rectangle2D.h"
#include "Matrix2D.h"
#include "Color.h"

namespace Torch {

Rectangle2D::Rectangle2D(Point2D P0_, int w_, int h_)
{
	P0 = P0_;
	P1 = P0_;
	P2 = P0_;
	P3 = P0_;

	P1.x += w_;
	P2.x += w_;
	P2.y += h_;
	P3.y += h_;
}

Rectangle2D::Rectangle2D(int x_, int y_, int w_, int h_)
{
	P0.reset(x_, y_);
	P1.reset(x_, y_);
	P2.reset(x_, y_);
	P3.reset(x_, y_);

	P1.x += w_;
	P2.x += w_;
	P2.y += h_;
	P3.y += h_;
}

Rectangle2D::Rectangle2D(sRect2D r)
{
	P0.reset(r.x, r.y);
	P1.reset(r.x, r.y);
	P2.reset(r.x, r.y);
	P3.reset(r.x, r.y);

	P1.x += r.w;
	P2.x += r.w;
	P2.y += r.h;
	P3.y += r.h;
}

void Rectangle2D::reset(Point2D P0_, Point2D P1_, Point2D P2_, Point2D P3_)
{
	P0 = P0_;
	P1 = P1_;
	P2 = P2_;
	P3 = P3_;
}

void Rectangle2D::reset(Point2D P0_, int w_, int h_)
{
	P0 = P0_;
	P1 = P0_;
	P2 = P0_;
	P3 = P0_;

	P1.x += w_;
	P2.x += w_;
	P2.y += h_;
	P3.y += h_;
}

void Rectangle2D::reset(int x_, int y_, int w_, int h_)
{
	P0.reset(x_, y_);
	P1.reset(x_, y_);
	P2.reset(x_, y_);
	P3.reset(x_, y_);

	P1.x += w_;
	P2.x += w_;
	P2.y += h_;
	P3.y += h_;
}

void Rectangle2D::reset(sRect2D r)
{
	P0.reset(r.x, r.y);
	P1.reset(r.x, r.y);
	P2.reset(r.x, r.y);
	P3.reset(r.x, r.y);

	P1.x += r.w;
	P2.x += r.w;
	P2.y += r.h;
	P3.y += r.h;
}

void Rectangle2D::saveFile(File *file)
{
   	P0.saveFile(file);
   	P1.saveFile(file);
   	P2.saveFile(file);
   	P3.saveFile(file);
}

void Rectangle2D::loadFile(File *file)
{
   	P0.loadFile(file);
   	P1.loadFile(file);
   	P2.loadFile(file);
   	P3.loadFile(file);
}

const char *Rectangle2D::sprint()
{
	sprintf(buf_sprint, "[(%g, %g) (%g, %g) (%g, %g) (%g, %g)]",
	      		P0.x, P0.y,
			P1.x, P1.y,
			P2.x, P2.y,
			P3.x, P3.y);

	return buf_sprint;
}

void Rectangle2D::draw(Image *image, Color color)
{
	image->drawLine((int)P0.x, (int)P0.y, (int)P1.x, (int)P1.y, color);
	image->drawLine((int)P1.x, (int)P1.y, (int)P2.x, (int)P2.y, color);
	image->drawLine((int)P2.x, (int)P2.y, (int)P3.x, (int)P3.y, color);
	image->drawLine((int)P3.x, (int)P3.y, (int)P0.x, (int)P0.y, color);
}

#ifdef HAVE_X11
void Rectangle2D::xdraw(Display *pDisplay, Pixmap pixmap, GC gc, unsigned long color, int line_width, int line_style, bool fill, int fill_style)
{
	XGCValues old_gc_values;
	XGCValues new_gc_values;

	int myChangeFlags = GCLineWidth|GCLineStyle|GCFillStyle;

	// Save current graphic context attributes
	XGetGCValues(pDisplay, gc, myChangeFlags, &old_gc_values);

	// Set new attributes
	new_gc_values.line_width = line_width;
	new_gc_values.line_style = line_style;
	new_gc_values.fill_style = fill_style;

	XChangeGC(pDisplay, gc, myChangeFlags, &new_gc_values);

	XSetForeground(pDisplay, gc, color);

	if(fill)
	{
	   	int w_ = (int) (P1.x - P0.x);
		int h_ = (int) (P2.y - P1.y);

		XFillRectangle(pDisplay, pixmap, gc, (int) P0.x, (int) P0.y, w_, h_);
	}
	else
	{
		XDrawLine(pDisplay, pixmap, gc, (int) P0.x, (int) P0.y, (int) P1.x, (int) P1.y);
		XDrawLine(pDisplay, pixmap, gc, (int) P1.x, (int) P1.y, (int) P2.x, (int) P2.y);
		XDrawLine(pDisplay, pixmap, gc, (int) P2.x, (int) P2.y, (int) P3.x, (int) P3.y);
		XDrawLine(pDisplay, pixmap, gc, (int) P3.x, (int) P3.y, (int) P0.x, (int) P0.y);
	}

	// Restore graphic context attributes
	XChangeGC(pDisplay, gc, myChangeFlags, &old_gc_values);
}
#endif

void Rectangle2D::fixI()
{
	P0.fixI();
	P1.fixI();
	P2.fixI();
	P3.fixI();
}

Rectangle2D Rectangle2D::operator+(Vector2D v)
{
	Rectangle2D R;

	R.P0 = P0 + v;
	R.P1 = P1 + v;
	R.P2 = P2 + v;
	R.P3 = P3 + v;

	return R;
}

Rectangle2D Rectangle2D::operator-(Vector2D v)
{
	Rectangle2D R;

	R.P0 = P0 - v;
	R.P1 = P1 - v;
	R.P2 = P2 - v;
	R.P3 = P3 - v;

	return R;
}

Rectangle2D operator*(Rectangle2D rs, Matrix2D m)
{
	Rectangle2D rd;

	rd.P0 = rs.P0 * m;
	rd.P1 = rs.P1 * m;
	rd.P2 = rs.P2 * m;
	rd.P3 = rs.P3 * m;

	return rd;
}

bool isInRect(int x_, int y_, sRect2D r_)
{
   	if((x_ >= r_.x) && (x_ <= r_.x+r_.w) && (y_ >= r_.y) && (y_ <= r_.y+r_.h)) return true;

	return false;
}

}
