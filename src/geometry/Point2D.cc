#include "Point2D.h"
#include "Vector2D.h"
#include "Matrix2D.h"
#include "Color.h"

namespace Torch {

//------------------------------------------------------------------
// IO streams
//------------------------------------------------------------------

// Write output Point2D
void Point2D::saveFile(File *file)
{
	file->write(&x, sizeof(double), 1);
	file->write(&y, sizeof(double), 1);
}

// Read input Point2D format
void Point2D::loadFile(File *file)
{
	file->read(&x, sizeof(double), 1);
	file->read(&y, sizeof(double), 1);
}

const char * Point2D::sprint()
{
	sprintf(buf_sprint, "(%g, %g)", x, y);

	return buf_sprint;
}

void Point2D::draw(Image *image, Color color)
{
   	image->drawPixel((int) x, (int) y, color);
}

//------------------------------------------------------------------
// Comparison
//------------------------------------------------------------------

int Point2D::operator==(Point2D Q)
{
	return (x==Q.x && y==Q.y);
}

int Point2D::operator!=(Point2D Q)
{
	return (x!=Q.x || y!=Q.y);
}

sPoint2D Point2D::get()
{
	sPoint2D sp;

	sp.x = x;
	sp.y = y;

	return sp;
}

//------------------------------------------------------------------
// Point2D Vector Operations
//------------------------------------------------------------------

Vector2D Point2D::operator-(Point2D Q)        // Vector diff of Points
{
	Vector2D v;
	v.x = x - Q.x;
	v.y = y - Q.y;
	return v;
}

Point2D Point2D::operator+(Vector2D v)        // +ve translation
{
	Point2D P;
	P.x = x + v.x;
	P.y = y + v.y;
	return P;
}

Point2D Point2D::operator-(Vector2D v)        // -ve translation
{
	Point2D P;
	P.x = x - v.x;
	P.y = y - v.y;
	return P;
}

Point2D& Point2D::operator+=(Vector2D v)        // +ve translation
{
	x += v.x;
	y += v.y;
	return *this;
}

Point2D& Point2D::operator-=(Vector2D v)        // -ve translation
{
	x -= v.x;
	y -= v.y;
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

Point2D operator*(int c, Point2D Q)
{
	Point2D P;
	P.x = c * Q.x;
	P.y = c * Q.y;
	return P;
}

Point2D operator*(double c, Point2D Q)
{
	Point2D P;
	P.x = c * Q.x;
	P.y = c * Q.y;
	return P;
}

Point2D operator*(Point2D Q, int c)
{
	Point2D P;
	P.x = c * Q.x;
	P.y = c * Q.y;
	return P;
}

Point2D operator*(Point2D Q, double c)
{
	Point2D P;
	P.x = c * Q.x;
	P.y = c * Q.y;
	return P;
}

Point2D operator/(Point2D Q, int c)
{
	Point2D P;
	P.x = Q.x / c;
	P.y = Q.y / c;
	return P;
}

Point2D operator/(Point2D Q, double c)
{
	Point2D P;
	P.x = Q.x / c;
	P.y = Q.y / c;
	return P;
}

//------------------------------------------------------------------
// Point2D Addition (also convenient but often illegal)
//    is not valid unless part of an affine sum.
//    The programmer must enforce this (if they want to).
//------------------------------------------------------------------

Point2D operator+(Point2D Q, Point2D R)
{
	Point2D P;
	P.x = Q.x + R.x;
	P.y = Q.y + R.y;
	return P;
}

Point2D operator*(Point2D x, Matrix2D m)
{
	Point2D w;

	w.x = m.ptr[0][0] * x.x + m.ptr[1][0] * x.y;
	w.y = m.ptr[0][1] * x.x + m.ptr[1][1] * x.y;

	return w;
}


//------------------------------------------------------------------
// Affine Sums
// Returns weighted sum, even when not affine, but...
// Tests if coeffs add to 1.
//------------------------------------------------------------------

Point2D asum(int n, int *c, Point2D *Q)
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
		P.x += c[i] * Q[i].x;
		P.y += c[i] * Q[i].y;
	}
	return P;
}

Point2D asum(int n, double *c, Point2D *Q)
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
		P.x += c[i] * Q[i].x;
		P.y += c[i] * Q[i].y;
	}
	return P;
}

//------------------------------------------------------------------
// Distance between Points
//------------------------------------------------------------------

double d(Point2D P, Point2D Q)
{      // Euclidean distance
	double dx = P.x - Q.x;
	double dy = P.y - Q.y;
	return sqrt(dx*dx + dy*dy);
}

double d2(Point2D P, Point2D Q)
{     // squared distance (more efficient)
	double dx = P.x - Q.x;
	double dy = P.y - Q.y;
	return (dx*dx + dy*dy);
}

//------------------------------------------------------------------
// Sidedness of a Point2D wrt a directed line P1->P2
//        - makes sense in 2D only
//------------------------------------------------------------------

double Point2D::isLeft(Point2D P1, Point2D P2)
{
	return ((P1.x - x) * (P2.y - y) - (P2.x - x) * (P1.y - y));
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
int isLeft(Point2D P0, Point2D P1, Point2D P2)
{
	return (int) ((P1.x - P0.x) * (P2.y - P0.y) - (P2.x - P0.x) * (P1.y - P0.y));
}

/** Get the angle (in radian) of a line (P1,P2) with the horizontal axis
*/
double angle(Point2D P1, Point2D P2)
{
   	if(P1.y == P2.y) return 0.0;

   	double deltax = (P2.x - P1.x);
	double deltay = (P2.y - P1.y);
	double r = sqrt(deltax*deltax + deltay*deltay);

	return M_PI_2 - asin(abs(deltax) / r);
	//return atan2(P2.y, P2.x) - atan2(P1.y, P1.x);
}

/** Get the angle (in radian) of 2 lines (P0,P1) and (P0,P2)
*/
double angle(Point2D P0, Point2D P1, Point2D P2)
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
int ccw(Point2D V0, Point2D V1, Point2D V2)
{
	return isLeft(V0, V1, V2);
}

void Point2D::fixI()
{
	int X_ = (int) (x + 0.5);
	int Y_ = (int) (y + 0.5);

	x = X_;
	y = Y_;
}

}
