#include "ip/Polygon2D.h"
#include "ip/Color.h"

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

void Polygon2D::draw(Image *image, Color color)
{
   	for(int i = 0 ; i < n-1 ; i++)
   		image->drawLine((int)V[i].get(0), (int)V[i].get(1),
				(int)V[i+1].get(0), (int)V[i+1].get(1), color);
}

/** winding number test for a point in a polygon
	Input:  P = a point,
		V[] = vertex points of a polygon V[n+1] with V[n]=V[0]
	Return: wn = the winding number (=0 only if P is outside V[])

	@author softSurfer (www.softsurfer.com)
	@see http://geometryalgorithms.com/Archive/algorithm_0103/algorithm_0103.htm
 */
int Polygon2D::wn_PnPoly(const Point2D& P_)
{
	int wn = 0;    // the winding number counter

	// loop through all edges of the polygon
	for (int i=0; i<n; i++)
	{ // edge from V[i] to V[i+1]
		if (V[i].get(1) <= P_.get(1))
		{ // start y <= P.y
			if (V[i+1].get(1) > P_.get(1))      // an upward crossing
				if (isLeft(V[i], V[i+1], P_) > 0)  // P left of edge
					++wn;            // have a valid up intersect
		}
		else
		{ // start y > P.y (no test needed)
			if (V[i+1].get(1) <= P_.get(1))     // a downward crossing
				if (isLeft(V[i], V[i+1], P_) < 0)  // P right of edge
					--wn;            // have a valid down intersect
		}
	}

	return wn;
}

}
