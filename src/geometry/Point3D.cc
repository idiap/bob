#include "Point3D.h"
#include "Vector3D.h"
#include "Color.h"

namespace Torch {

//------------------------------------------------------------------
// IO streams
//------------------------------------------------------------------

// Write output Point3D
void Point3D::saveFile(File *file)
{
	file->write(&x, sizeof(double), 1);
	file->write(&y, sizeof(double), 1);
	file->write(&z, sizeof(double), 1);
}

// Read input Point3D format
void Point3D::loadFile(File *file)
{
	file->read(&x, sizeof(double), 1);
	file->read(&y, sizeof(double), 1);
	file->read(&z, sizeof(double), 1);
}

const char * Point3D::sprint()
{
	sprintf(buf_sprint, "(%g, %g, %g)", x, y, z);

	return buf_sprint;
}

void Point3D::draw(Image *image, Color color)
{
   	message("Don't know how to draw in 3D.");
}

//------------------------------------------------------------------
// Comparison
//------------------------------------------------------------------

int Point3D::operator==(Point3D Q)
{
	return (x==Q.x && y==Q.y && z==Q.z);
}

int Point3D::operator!=(Point3D Q)
{
	return (x!=Q.x || y!=Q.y || z!=Q.z);
}

//------------------------------------------------------------------
// Point3D Vector3D Operations
//------------------------------------------------------------------

Vector3D Point3D::operator-(Point3D Q)        // Vector diff of Points
{
	Vector3D v;
	v.x = x - Q.x;
	v.y = y - Q.y;
	v.z = z - Q.z;
	return v;
}

Point3D Point3D::operator+(Vector3D v)        // +ve translation
{
	Point3D P;
	P.x = x + v.x;
	P.y = y + v.y;
	P.z = z + v.z;
	return P;
}

Point3D Point3D::operator-(Vector3D v)        // -ve translation
{
	Point3D P;
	P.x = x - v.x;
	P.y = y - v.y;
	P.z = z - v.z;
	return P;
}

Point3D& Point3D::operator+=(Vector3D v)        // +ve translation
{
	x += v.x;
	y += v.y;
	z += v.z;
	return *this;
}

Point3D& Point3D::operator-=(Vector3D v)        // -ve translation
{
	x -= v.x;
	y -= v.y;
	z -= v.z;
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

Point3D operator*(int c, Point3D Q)
{
	Point3D P;
	P.x = c * Q.x;
	P.y = c * Q.y;
	P.z = c * Q.z;
	return P;
}

Point3D operator*(double c, Point3D Q)
{
	Point3D P;
	P.x = c * Q.x;
	P.y = c * Q.y;
	P.z = c * Q.z;
	return P;
}

Point3D operator*(Point3D Q, int c)
{
	Point3D P;
	P.x = c * Q.x;
	P.y = c * Q.y;
	P.z = c * Q.z;
	return P;
}

Point3D operator*(Point3D Q, double c)
{
	Point3D P;
	P.x = c * Q.x;
	P.y = c * Q.y;
	P.z = c * Q.z;
	return P;
}

Point3D operator/(Point3D Q, int c)
{
	Point3D P;
	P.x = Q.x / c;
	P.y = Q.y / c;
	P.z = Q.z / c;
	return P;
}

Point3D operator/(Point3D Q, double c)
{
	Point3D P;
	P.x = Q.x / c;
	P.y = Q.y / c;
	P.z = Q.z / c;
	return P;
}

//------------------------------------------------------------------
// Point3D Addition (also convenient but often illegal)
//    is not valid unless part of an affine sum.
//    The programmer must enforce this (if they want to).
//------------------------------------------------------------------

Point3D operator+(Point3D Q, Point3D R)
{
	Point3D P;
	P.x = Q.x + R.x;
	P.y = Q.y + R.y;
	P.z = Q.z + R.z;
	return P;
}

//------------------------------------------------------------------
// Affine Sums
// Returns weighted sum, even when not affine, but...
// Tests if coeffs add to 1.  If not, sets: err = Esum.
//------------------------------------------------------------------

Point3D asum(int n, int *c, Point3D *Q)
{
	int        cs = 0;
	Point3D      P;

	for (int i=0; i<n; i++)
		cs += c[i];
	if (cs != 1)                 // not an affine sum
		warning("Sum not affine.");

	for (int i=0; i<n; i++)
	{
		P.x += c[i] * Q[i].x;
		P.y += c[i] * Q[i].y;
		P.z += c[i] * Q[i].z;
	}
	return P;
}

Point3D asum(int n, double *c, Point3D *Q)
{
	double     cs = 0.0;
	Point3D      P;

	for (int i=0; i<n; i++)
		cs += c[i];
	if (cs != 1)                 // not an affine sum
		warning("Sum not affine.");

	for (int i=0; i<n; i++)
	{
		P.x += c[i] * Q[i].x;
		P.y += c[i] * Q[i].y;
		P.z += c[i] * Q[i].z;
	}
	return P;
}

//------------------------------------------------------------------
// Distance between Points
//------------------------------------------------------------------

double d(Point3D P, Point3D Q)
{      // Euclidean distance
	double dx = P.x - Q.x;
	double dy = P.y - Q.y;
	double dz = P.z - Q.z;
	return sqrt(dx*dx + dy*dy + dz*dz);
}

double d2(Point3D P, Point3D Q)
{     // squared distance (more efficient)
	double dx = P.x - Q.x;
	double dy = P.y - Q.y;
	double dz = P.z - Q.z;
	return (dx*dx + dy*dy + dz*dz);
}

}
