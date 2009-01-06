#include "Polygon2D.h"
#include "Color.h"

namespace Torch {

Polygon2D::Polygon2D(int n_)
{
   	n = n_;
	V = new Point2D [n+2];
	is_allocated = true;
}

Polygon2D::Polygon2D(int n_, Point2D *V_)
{
   	n = n_;
	V = V_;
	is_allocated = false;
}

Polygon2D::~Polygon2D()
{
	if(is_allocated == true)
   		delete [] V;
}

void Polygon2D::saveFile(File *file)
{
   	file->write(&n, sizeof(int), 1);
	for(int i = 0 ; i < n ; i++) V[i].saveFile(file);
}

void Polygon2D::loadFile(File *file)
{
   	file->read(&n, sizeof(int), 1);

	for(int i = 0 ; i < n ; i++) V[i].loadFile(file);
}

const char *Polygon2D::sprint()
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

void Polygon2D::draw(Image *image, Color color)
{
   	for(int i = 0 ; i < n-1 ; i++)
		image->drawLine((int)V[i].x, (int)V[i].y, (int)V[i+1].x, (int)V[i+1].y, color);
}

/** winding number test for a point in a polygon
	Input:  P = a point,
		V[] = vertex points of a polygon V[n+1] with V[n]=V[0]
	Return: wn = the winding number (=0 only if P is outside V[])

	@author softSurfer (www.softsurfer.com)
	@see http://geometryalgorithms.com/Archive/algorithm_0103/algorithm_0103.htm
 */
int Polygon2D::wn_PnPoly(Point2D P_)
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
