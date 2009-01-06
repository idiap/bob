#include "Circle2D.h"
#include "Color.h"

namespace Torch {

Circle2D::Circle2D()
{
   	n = 0;
	V = NULL;
}

Circle2D::Circle2D(sPoint2D center_, double radius_)
{
	double step = 0.5;

	n = (int) (360 / step);
	V = new Point2D [n+2];

	//
	radius = radius_;
	center = center_;

	double d_angle = 0.0;
	double r_angle;

	int i = 0;

	V[i++].reset(center.x + radius, center.y);
	d_angle += step;

	r_angle = degree2radian(d_angle);
	V[i++].reset(center.x + radius * cos(r_angle), center.y + radius * sin(r_angle));
	while(d_angle <= 360.0)
	{
		r_angle = degree2radian(d_angle);
		V[i++].reset(center.x + radius * cos(r_angle), center.y + radius * sin(r_angle));
		d_angle += step;
	}
}

Circle2D::Circle2D(int xc_, int yc_, double radius_)
{
	double step = 0.5;

	n = (int) (360 / step);
	V = new Point2D [n+2];

	//
	radius = radius_;
	center.x = center.x;
	center.y = center.y;

	double d_angle = 0.0;
	double r_angle;

	int i = 0;

	V[i++].reset(center.x + radius, center.y);
	d_angle += step;

	r_angle = degree2radian(d_angle);
	V[i++].reset(center.x + radius * cos(r_angle), center.y + radius * sin(r_angle));
	while(d_angle <= 360.0)
	{
		r_angle = degree2radian(d_angle);
		V[i++].reset(center.x + radius * cos(r_angle), center.y + radius * sin(r_angle));
		d_angle += step;
	}
}

void Circle2D::reset(sPoint2D center_, double radius_)
{
	double step = 0.5;

	n = (int) (360 / step);
	if(V != NULL) delete [] V;
	V = new Point2D [n+2];

	//
	radius = radius_;
	center = center_;

	double d_angle = 0.0;
	double r_angle;

	int i = 0;

	V[i++].reset(center.x + radius, center.y);
	d_angle += step;

	r_angle = degree2radian(d_angle);
	V[i++].reset(center.x + radius * cos(r_angle), center.y + radius * sin(r_angle));
	while(d_angle <= 360.0)
	{
		r_angle = degree2radian(d_angle);
		V[i++].reset(center.x + radius * cos(r_angle), center.y + radius * sin(r_angle));
		d_angle += step;
	}
}

void Circle2D::reset(int xc_, int yc_, double radius_)
{
	double step = 0.5;

	n = (int) (360 / step);
	if(V != NULL) delete [] V;
	V = new Point2D [n+2];

	//
	radius = radius_;
	center.x = xc_;
	center.y = yc_;

	double d_angle = 0.0;
	double r_angle;

	int i = 0;

	V[i++].reset(center.x + radius, center.y);
	d_angle += step;

	r_angle = degree2radian(d_angle);
	V[i++].reset(center.x + radius * cos(r_angle), center.y + radius * sin(r_angle));
	while(d_angle <= 360.0)
	{
		r_angle = degree2radian(d_angle);
		V[i++].reset(center.x + radius * cos(r_angle), center.y + radius * sin(r_angle));
		d_angle += step;
	}
}

Circle2D::~Circle2D()
{
	if(V != NULL) delete [] V;
}

void Circle2D::saveFile(File *file)
{
   	file->write(&n, sizeof(int), 1);
	for(int i = 0 ; i < n ; i++) V[i].saveFile(file);
}

void Circle2D::loadFile(File *file)
{
   	file->read(&n, sizeof(int), 1);

	for(int i = 0 ; i < n ; i++) V[i].loadFile(file);
}

const char *Circle2D::sprint()
{
   	char str_[250];

	strcpy(buf_sprint, "{ ");

	for(int i = 0 ; i < n ; i++)
	{
		sprintf(str_, "(%g, %g) " , V[i].x, V[i].y);
		strcat(buf_sprint, str_);
	}

	strcat(buf_sprint, "}");

	return buf_sprint;
}

void Circle2D::draw(Image *image, Color color)
{
   	for(int i = 0 ; i < n-1 ; i++)
		image->drawLine((int)V[i].x, (int)V[i].y, (int)V[i+1].x, (int)V[i+1].y, color);
}

#ifdef HAVE_X11
void Circle2D::xdraw(Display *pDisplay, Pixmap pixmap, GC gc, unsigned long color, int line_width, int line_style, bool fill, int fill_style)
{
	int diameter = (int) (radius*2);
	int x = (int) (center.x - radius);
	int y = (int) (center.y - radius);

	XSetForeground(pDisplay, gc, color);

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

	XDrawArc(pDisplay, pixmap, gc, x, y, diameter, diameter, 0, 360*64);
   	//for(int i = 0 ; i < n-1 ; i++) XDrawLine(pDisplay, pixmap, gc, (int) V[i].x, (int) V[i].y, (int) V[i+1].x, (int) V[i+1].y);

	// Restore graphic context attributes
	XChangeGC(pDisplay, gc, myChangeFlags, &old_gc_values);
}
#endif

/** winding number test for a point in a polygon
	Input:  P = a point,
		V[] = vertex points of a polygon V[n+1] with V[n]=V[0]
	Return: wn = the winding number (=0 only if P is outside V[])

	@author softSurfer (www.softsurfer.com)
	@see http://geometryalgorithms.com/Archive/algorithm_0103/algorithm_0103.htm
 */
int Circle2D::wn_PnPoly(Point2D P_)
{
	int wn = 0;    // the winding number counter

	// loop through all edges of the polygon
	for (int i=0; i<n; i++)
	{ // edge from V[i] to V[i+1]
		if (V[i].y <= P_.y)
		{ // start y <= P.y
			if (V[i+1].y > P_.y)      // an upward crossing
				if (isLeft(V[i], V[i+1], P_) > 0)  // P left of edge
					++wn;            // have a valid up intersect
		}
		else
		{ // start y > P.y (no test needed)
			if (V[i+1].y <= P_.y)     // a downward crossing
				if (isLeft(V[i], V[i+1], P_) < 0)  // P right of edge
					--wn;            // have a valid down intersect
		}
	}

	return wn;
}

}
