#include "ip/Point3D.h"
#include "ip/Vector3D.h"
#include "ip/OldColor.h"

namespace Torch {

//------------------------------------------------------------------
// Constructors

Point3D::Point3D() : DoubleTensor(3)
{
	fill(0.0);
}

Point3D::Point3D(int a) : DoubleTensor(3)
{
	set(0, a);
	set(1, a);
	set(2, 0.0);
}

Point3D::Point3D(double a) : DoubleTensor(3)
{
	set(0, a);
	set(1, a);
	set(2, 0.0);
}

Point3D::Point3D(int a, int b) : DoubleTensor(3)
{
	set(0, a);
	set(1, b);
	set(2, 0.0);
}

Point3D::Point3D(double a, double b) : DoubleTensor(3)
{
	set(0, a);
	set(1, b);
	set(2, 0.0);
}

Point3D::Point3D(int a, int b, int c) : DoubleTensor(3)
{
	set(0, a);
	set(1, b);
	set(2, c);
}

Point3D::Point3D(double a, double b, double c) : DoubleTensor(3)
{
	set(0, a);
	set(1, b);
	set(2, c);
}

void Point3D::draw(Image *image, Color color)
{
   	message("Don't know how to draw in 3D.");
}

//------------------------------------------------------------------
// Comparison
//------------------------------------------------------------------

int Point3D::operator==(const Point3D& Q)
{
	return (get(0)==Q.get(0) && get(1)==Q.get(1) && get(2)==Q.get(2));
}

int Point3D::operator!=(const Point3D& Q)
{
	return (get(0)!=Q.get(0) || get(1)!=Q.get(1) || get(2)!=Q.get(2));
}

//------------------------------------------------------------------
// Point3D Vector3D Operations
//------------------------------------------------------------------

Vector3D Point3D::operator-(const Point3D& Q)        // Vector diff of Points
{
	return Vector3D(get(0) - Q.get(0),
			get(1) - Q.get(1),
			get(2) - Q.get(2));
}

Point3D Point3D::operator+(const Vector3D& v)        // +ve translation
{
	return Point3D(	get(0) + v.get(0),
			get(1) + v.get(1),
			get(2) + v.get(2));
}

Point3D Point3D::operator-(const Vector3D& v)        // -ve translation
{
	return Point3D(	get(0) - v.get(0),
			get(1) - v.get(1),
			get(2) - v.get(2));
}

Point3D& Point3D::operator+=(const Vector3D& v)        // +ve translation
{
	set(0, get(0) + v.get(0));
	set(1, get(1) + v.get(1));
	set(2, get(2) + v.get(2));
	return *this;
}

Point3D& Point3D::operator-=(const Vector3D& v)        // -ve translation
{
	set(0, get(0) - v.get(0));
	set(1, get(1) - v.get(1));
	set(2, get(2) - v.get(2));
	return *this;
}

//------------------------------------------------------------------
// Point3D Scalar Operations (convenient but often illegal)
//        are not valid for points in general,
//        unless they are 'affine' as coeffs of
//        a sum in which all the coeffs add to 1,
//        such as: the sum (a*P + b*Q) with (a+b == 1).
//        The programmer must enforce this (if they want to).
//------------------------------------------------------------------

Point3D operator*(int c, const Point3D& Q)
{
	return Point3D(	c * Q.get(0),
			c * Q.get(1),
			c * Q.get(2));
}

Point3D operator*(double c, const Point3D& Q)
{
	return Point3D(	c * Q.get(0),
			c * Q.get(1),
			c * Q.get(2));
}

Point3D operator*(const Point3D& Q, int c)
{
	return Point3D(	c * Q.get(0),
			c * Q.get(1),
			c * Q.get(2));
}

Point3D operator*(const Point3D& Q, double c)
{
	return Point3D(	c * Q.get(0),
			c * Q.get(1),
			c * Q.get(2));
}

Point3D operator/(const Point3D& Q, int c)
{
	return Point3D(	Q.get(0) / c,
			Q.get(1) / c,
			Q.get(2) / c);
}

Point3D operator/(const Point3D& Q, double c)
{
	return Point3D(	Q.get(0) / c,
			Q.get(1) / c,
			Q.get(2) / c);
}

//------------------------------------------------------------------
// Point3D Addition (also convenient but often illegal)
//    is not valid unless part of an affine sum.
//    The programmer must enforce this (if they want to).
//------------------------------------------------------------------

Point3D operator+(const Point3D& Q, const Point3D& R)
{
	return Point3D(	Q.get(0) + R.get(0),
			Q.get(1) + R.get(1),
			Q.get(2) + R.get(2));
}

//------------------------------------------------------------------
// Affine Sums
// Returns weighted sum, even when not affine, but...
// Tests if coeffs add to 1.  If not, sets: err = Esum.
//------------------------------------------------------------------

Point3D asum(int n, int *c, const Point3D *Q)
{
	int        cs = 0;
	Point3D      P;

	for (int i=0; i<n; i++)
		cs += c[i];
	if (cs != 1)                 // not an affine sum
		warning("Sum not affine.");

	for (int i=0; i<n; i++)
	{
		P.set(0, P.get(0) + c[i] * Q[i].get(0));
		P.set(1, P.get(1) + c[i] * Q[i].get(1));
		P.set(2, P.get(2) + c[i] * Q[i].get(2));
	}
	return P;
}

Point3D asum(int n, double *c, const Point3D *Q)
{
	double     cs = 0.0;
	Point3D      P;

	for (int i=0; i<n; i++)
		cs += c[i];
	if (cs != 1)                 // not an affine sum
		warning("Sum not affine.");

	for (int i=0; i<n; i++)
	{
		P.set(0, P.get(0) + c[i] * Q[i].get(0));
		P.set(1, P.get(1) + c[i] * Q[i].get(1));
		P.set(2, P.get(2) + c[i] * Q[i].get(2));
	}
	return P;
}

//------------------------------------------------------------------
// Distance between Points
//------------------------------------------------------------------

double d(const Point3D& P, const Point3D& Q)
{      // Euclidean distance
	double dx = P.get(0) - Q.get(0);
	double dy = P.get(1) - Q.get(1);
	double dz = P.get(2) - Q.get(2);
	return sqrt(dx*dx + dy*dy + dz*dz);
}

double d2(const Point3D& P, const Point3D& Q)
{     // squared distance (more efficient)
	double dx = P.get(0) - Q.get(0);
	double dy = P.get(1) - Q.get(1);
	double dz = P.get(2) - Q.get(2);
	return (dx*dx + dy*dy + dz*dz);
}

}
