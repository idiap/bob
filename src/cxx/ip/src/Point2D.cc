#include "ip/Point2D.h"
#include "ip/Vector2D.h"
#include "ip/Matrix2D.h"
#include "ip/OldColor.h"

namespace Torch {

//------------------------------------------------------------------
// Constructors

Point2D::Point2D() : DoubleTensor(2)
{
	fill(0.0);
}

Point2D::Point2D(int a) : DoubleTensor(2)
{
	fill(a);
}

Point2D::Point2D(double a) : DoubleTensor(2)
{
	fill(a);
}

Point2D::Point2D(int a, int b) : DoubleTensor(2)
{
	set(0, a);
	set(1, b);
}

Point2D::Point2D(double a, double b) : DoubleTensor(2)
{
	set(0, a);
	set(1, b);
}

Point2D::Point2D(const sPoint2D& p) : DoubleTensor(2)
{
	set(0, p.x);
	set(1, p.y);
}

Point2D::Point2D(const sPoint2D* p) : DoubleTensor(2)
{
	set(0, p->x);
	set(1, p->y);
}

void Point2D::draw(Image *image, Color color)
{
   	image->drawPixel((int) get(0), (int) get(1), color);
}

//------------------------------------------------------------------
// Comparison
//------------------------------------------------------------------

int Point2D::operator==(const Point2D& Q)
{
	return (get(0)==Q.get(0) && get(1)==Q.get(1));
}

int Point2D::operator!=(const Point2D& Q)
{
	return (get(0)!=Q.get(0) || get(1)!=Q.get(1));
}

sPoint2D Point2D::getPoint()
{
	return sPoint2D(get(0), get(1));
}

//------------------------------------------------------------------
// Point2D Vector Operations
//------------------------------------------------------------------

Vector2D Point2D::operator-(const Point2D& Q)        // Vector diff of Points
{
	return Vector2D(get(0) - Q.get(0),
			get(1) - Q.get(1));
}

Point2D Point2D::operator+(const Vector2D& v)        // +ve translation
{
	return Vector2D(get(0) + v.get(0),
			get(1) + v.get(1));
}

Point2D Point2D::operator-(const Vector2D& v)        // -ve translation
{
	return Vector2D(get(0) - v.get(0),
			get(1) - v.get(1));
}

Point2D& Point2D::operator+=(const Vector2D& v)        // +ve translation
{
	set(0, get(0) + v.get(0));
	set(1, get(1) + v.get(1));
	return *this;
}

Point2D& Point2D::operator-=(const Vector2D& v)        // -ve translation
{
	set(0, get(0) - v.get(0));
	set(1, get(1) - v.get(1));
	return *this;
}

//------------------------------------------------------------------
// Point2D Scalar Operations (convenient but often illegal)
//        are not valid for points in general,
//        unless they are 'affine' as coeffs of
//        a sum in which all the coeffs add to 1,
//        such as: the sum (a*P + b*Q) with (a+b == 1).
//        The programmer must enforce this (if they want to).
//------------------------------------------------------------------

Point2D operator*(int c, const Point2D& Q)
{
	return Point2D(	c * Q.get(0),
			c * Q.get(1));
}

Point2D operator*(double c, const Point2D& Q)
{
	return Point2D(	c * Q.get(0),
			c * Q.get(1));
}

Point2D operator*(const Point2D& Q, int c)
{
	return Point2D(	c * Q.get(0),
			c * Q.get(1));
}

Point2D operator*(const Point2D& Q, double c)
{
	return Point2D(	c * Q.get(0),
			c * Q.get(1));
}

Point2D operator/(const Point2D& Q, int c)
{
	return Point2D(	Q.get(0) / c,
			Q.get(1) / c);
}

Point2D operator/(const Point2D& Q, double c)
{
	return Point2D(	Q.get(0) / c,
			Q.get(1) / c);
}

//------------------------------------------------------------------
// Point2D Addition (also convenient but often illegal)
//    is not valid unless part of an affine sum.
//    The programmer must enforce this (if they want to).
//------------------------------------------------------------------

Point2D operator+(const Point2D& Q, const Point2D& R)
{
	return Point2D(	Q.get(0) + R.get(0),
			Q.get(1) + R.get(1));
}

Point2D operator*(const Point2D& x, const Matrix2D& m)
{
	return Point2D(	m.get(0, 0) * x.get(0) + m.get(1, 0) * x.get(1),
			m.get(0, 1) * x.get(0) + m.get(1, 1) * x.get(1));
}


//------------------------------------------------------------------
// Affine Sums
// Returns weighted sum, even when not affine, but...
// Tests if coeffs add to 1.
//------------------------------------------------------------------

Point2D asum(int n, int *c, const Point2D *Q)
{
	int        cs = 0;
	Point2D      P;

	for (int i=0; i<n; i++)
	{
		cs += c[i];
	}
	if (cs != 1)                 // not an affine sum
		warning("Not affine sum.\n");

	for (int i=0; i<n; i++)
	{
		P.set(0, P.get(0) + c[i] * Q[i].get(0));
		P.set(1, P.get(1) + c[i] * Q[i].get(1));
	}
	return P;
}

Point2D asum(int n, double *c, const Point2D *Q)
{
	double     cs = 0.0;
	Point2D      P;

	for (int i=0; i<n; i++)
	{
		cs += c[i];
	}
	if (cs != 1)                 // not an affine sum
		warning("Not affine sum.\n");

	for (int i=0; i<n; i++)
	{
		P.set(0, P.get(0) + c[i] * Q[i].get(0));
		P.set(1, P.get(1) + c[i] * Q[i].get(1));
	}
	return P;
}

//------------------------------------------------------------------
// Distance between Points
//------------------------------------------------------------------

double d(const Point2D& P, const Point2D& Q)
{      // Euclidean distance
	const double dx = P.get(0) - Q.get(0);
	const double dy = P.get(1) - Q.get(1);
	return sqrt(dx*dx + dy*dy);
}

double d2(const Point2D& P, const Point2D& Q)
{     // squared distance (more efficient)
	const double dx = P.get(0) - Q.get(0);
	const double dy = P.get(1) - Q.get(1);
	return (dx*dx + dy*dy);
}

//------------------------------------------------------------------
// Sidedness of a Point2D wrt a directed line P1->P2
//        - makes sense in 2D only
//------------------------------------------------------------------

double Point2D::isLeft(const Point2D& P1, const Point2D& P2)
{
	return 	(P1.get(0) - get(0)) * (P2.get(1) - get(1)) -
		(P2.get(0) - get(0)) * (P1.get(1) - get(1));
}

/** isLeft(): tests if a point is Left|On|Right of an infinite line
	Input:  three points P0, P1, and P2
	Return: >0 for P2 left of the line through P0 and P1
		=0 for P2 on the line
		<0 for P2 right of the line

	@author softSurfer (www.softsurfer.com)
	@see http://geometryalgorithms.com/Archive/algorithm_0103/algorithm_0103.htm
	@see Point2D::isLeft(Point2D P1, Point2D P2)

*/
int isLeft(const Point2D& P0, const Point2D& P1, const Point2D& P2)
{
	return 	(P1.get(0) - P0.get(0)) * (P2.get(1) - P0.get(1)) -
		(P2.get(0) - P0.get(0)) * (P1.get(1) - P0.get(1));
}

/** Get the angle (in radian) of a line (P1,P2) with the horizontal axis
*/
double angle(const Point2D& P1, const Point2D& P2)
{
   	if(P1.get(1) == P2.get(1)) return 0.0;

   	double deltax = (P2.get(0) - P1.get(0));
	double deltay = (P2.get(1) - P1.get(1));
	double r = sqrt(deltax*deltax + deltay*deltay);

	return M_PI_2 - asin(abs(deltax) / r);
	//return atan2(P2.y, P2.x) - atan2(P1.y, P1.x);
}

/** Get the angle (in radian) of 2 lines (P0,P1) and (P0,P2)
*/
double angle(const Point2D& P0, const Point2D& P1, const Point2D& P2)
{
	double alpha = d(P0, P1);
	double beta = d(P1, P2);
	double gamma = d(P0, P2);

	return acos((alpha * alpha + gamma * gamma - beta * beta) / (2.0 * alpha * gamma));
}

/** Test the orientation 2D of a triangle
      Input:  three vertex points V0, V1, V2
      Return: angle direction of vectors (V0,V1) and (V0,V2)
      	      >0 for counterclockwise (ccw)
              =0 for none (degenerate)
              <0 for clockwise
*/
int ccw(const Point2D& V0, const Point2D& V1, const Point2D& V2)
{
	return isLeft(V0, V1, V2);
}

void Point2D::fixI()
{
	set(0, FixI(get(0)));
	set(1, FixI(get(1)));
}

}
