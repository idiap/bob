#include "ip/Rectangle2D.h"
#include "ip/Matrix2D.h"
#include "ip/Color.h"

namespace Torch {

Rectangle2D::Rectangle2D(const Point2D& P0_, int w_, int h_)
{
	P0 = P0_;
	P1 = P0_;
	P2 = P0_;
	P3 = P0_;

	P1.set(0, P1.get(0) + w_);
	P2.set(0, P2.get(0) + w_);
	P2.set(1, P2.get(1) + h_);
	P3.set(1, P3.get(1) + h_);
}

Rectangle2D::Rectangle2D(int x_, int y_, int w_, int h_)
{
	P0.reset(x_, y_);
	P1.reset(x_, y_);
	P2.reset(x_, y_);
	P3.reset(x_, y_);

	P1.set(0, P1.get(0) + w_);
	P2.set(0, P2.get(0) + w_);
	P2.set(1, P2.get(1) + h_);
	P3.set(1, P3.get(1) + h_);
}

Rectangle2D::Rectangle2D(const sRect2D& r)
{
	P0.reset(r.x, r.y);
	P1.reset(r.x, r.y);
	P2.reset(r.x, r.y);
	P3.reset(r.x, r.y);

	P1.set(0, P1.get(0) + r.w);
	P2.set(0, P2.get(0) + r.w);
	P2.set(1, P2.get(1) + r.h);
	P3.set(1, P3.get(1) + r.h);
}

void Rectangle2D::reset(const Point2D& P0_, const Point2D& P1_, const Point2D& P2_, const Point2D& P3_)
{
	P0 = P0_;
	P1 = P1_;
	P2 = P2_;
	P3 = P3_;
}

void Rectangle2D::reset(const Point2D& P0_, int w_, int h_)
{
	P0 = P0_;
	P1 = P0_;
	P2 = P0_;
	P3 = P0_;

	P1.set(0, P1.get(0) + w_);
	P2.set(0, P2.get(0) + w_);
	P2.set(1, P2.get(1) + h_);
	P3.set(1, P3.get(1) + h_);
}

void Rectangle2D::reset(int x_, int y_, int w_, int h_)
{
	P0.reset(x_, y_);
	P1.reset(x_, y_);
	P2.reset(x_, y_);
	P3.reset(x_, y_);

	P1.set(0, P1.get(0) + w_);
	P2.set(0, P2.get(0) + w_);
	P2.set(1, P2.get(1) + h_);
	P3.set(1, P3.get(1) + h_);
}

void Rectangle2D::reset(const sRect2D& r)
{
	P0.reset(r.x, r.y);
	P1.reset(r.x, r.y);
	P2.reset(r.x, r.y);
	P3.reset(r.x, r.y);

	P1.set(0, P1.get(0) + r.w);
	P2.set(0, P2.get(0) + r.w);
	P2.set(1, P2.get(1) + r.h);
	P3.set(1, P3.get(1) + r.h);
}

void Rectangle2D::draw(Image *image, Color color)
{
	image->drawLine((int)P0.get(0), (int)P0.get(1), (int)P1.get(0), (int)P1.get(1), color);
	image->drawLine((int)P1.get(0), (int)P1.get(1), (int)P2.get(0), (int)P2.get(1), color);
	image->drawLine((int)P2.get(0), (int)P2.get(1), (int)P3.get(0), (int)P3.get(1), color);
	image->drawLine((int)P3.get(0), (int)P3.get(1), (int)P0.get(0), (int)P0.get(1), color);
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

Rectangle2D Rectangle2D::operator+(const Vector2D& v)
{
	return Rectangle2D(	P0 + v,
				P1 + v,
				P2 + v,
				P3 + v);
}

Rectangle2D Rectangle2D::operator-(const Vector2D& v)
{
	return Rectangle2D(	P0 - v,
				P1 - v,
				P2 - v,
				P3 - v);
}

Rectangle2D operator*(const Rectangle2D& rs, const Matrix2D& m)
{
	return Rectangle2D(	rs.P0 * m,
				rs.P1 * m,
				rs.P2 * m,
				rs.P3 * m);
}

bool isInRect(int x_, int y_, const sRect2D& r_)
{
   	if((x_ >= r_.x) && (x_ <= r_.x+r_.w) && (y_ >= r_.y) && (y_ <= r_.y+r_.h)) return true;

	return false;
}

}
