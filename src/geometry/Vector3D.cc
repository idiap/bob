#include "Vector3D.h"

namespace Torch {

//------------------------------------------------------------------
//  Unary Ops
//------------------------------------------------------------------

// Unary minus
Vector3D Vector3D::operator-() 
{
	Vector3D v;
	v.x = -x; v.y = -y; v.z = -z;
	return v;
}

// Unary 2D perp operator
Vector3D Vector3D::operator~() 
{
	Vector3D v;
	v.x = -y; v.y = x; v.z = z;
	return v;
}

//------------------------------------------------------------------
//  Scalar Ops
//------------------------------------------------------------------

// Scalar multiplication
Vector3D operator*(int c, Vector3D w) 
{
	Vector3D v;
	v.x = c * w.x;
	v.y = c * w.y;
	v.z = c * w.z;
	return v;
}

Vector3D operator*(double c, Vector3D w) 
{
	Vector3D v;
	v.x = c * w.x;
	v.y = c * w.y;
	v.z = c * w.z;
	return v;
}

Vector3D operator*(Vector3D w, int c) 
{
	Vector3D v;
	v.x = c * w.x;
	v.y = c * w.y;
	v.z = c * w.z;
	return v;
}

Vector3D operator*(Vector3D w, double c) 
{
	Vector3D v;
	v.x = c * w.x;
	v.y = c * w.y;
	v.z = c * w.z;
	return v;
}

// Scalar division
Vector3D operator/(Vector3D w, int c) 
{
	Vector3D v;
	v.x = w.x / c;
	v.y = w.y / c;
	v.z = w.z / c;
	return v;
}

Vector3D operator/(Vector3D w, double c) 
{
	Vector3D v;
	v.x = w.x / c;
	v.y = w.y / c;
	v.z = w.z / c;
	return v;
}

//------------------------------------------------------------------
//  Arithmetic Ops
//------------------------------------------------------------------

Vector3D Vector3D::operator+(Vector3D w) 
{
	Vector3D v;
	v.x = x + w.x;
	v.y = y + w.y;
	v.z = z + w.z;
	return v;
}

Vector3D Vector3D::operator-(Vector3D w) 
{
	Vector3D v;
	v.x = x - w.x;
	v.y = y - w.y;
	v.z = z - w.z;
	return v;
}

//------------------------------------------------------------------
//  Products
//------------------------------------------------------------------

// Inner Dot Product
double Vector3D::operator*(Vector3D w) 
{
	return (x * w.x + y * w.y + z * w.z);
}

// 2D Exterior Perp Product
double Vector3D::operator|(Vector3D w) 
{
	return (x * w.y - y * w.x);
}

// 3D Exterior Cross Product
Vector3D Vector3D::operator^(Vector3D w) 
{
	Vector3D v;
	v.x = y * w.z - z * w.y;
	v.y = z * w.x - x * w.z;
	v.z = x * w.y - y * w.x;
	return v;
}

//------------------------------------------------------------------
//  Shorthand Ops
//------------------------------------------------------------------

Vector3D& Vector3D::operator*=(double c) 
{        // vector scalar mult
	x *= c;
	y *= c;
	z *= c;
	return *this;
}

Vector3D& Vector3D::operator/=(double c) 
{        // vector scalar div
	x /= c;
	y /= c;
	z /= c;
	return *this;
}

Vector3D& Vector3D::operator+=(Vector3D w) 
{        // vector increment
	x += w.x;
	y += w.y;
	z += w.z;
	return *this;
}

Vector3D& Vector3D::operator-=(Vector3D w) 
{        // vector decrement
	x -= w.x;
	y -= w.y;
	z -= w.z;
	return *this;
}

Vector3D& Vector3D::operator^=(Vector3D w) 
{        // 3D exterior cross product
	double ox=x, oy=y, oz=z;
	x = oy * w.z - oz * w.y;
	y = oz * w.x - ox * w.z;
	z = ox * w.y - oy * w.x;
	return *this;
}

//------------------------------------------------------------------
//  Special Operations
//------------------------------------------------------------------

void Vector3D::normalize() 
{                      // convert to unit length
	double ln = sqrt(x*x + y*y + z*z);
	if (ln == 0) return;                    // do nothing for nothing
	x /= ln;
	y /= ln;
	z /= ln;
}

Vector3D sum(int n, int *c, Vector3D *w) 
{     // vector sum
	Vector3D  v;

	for (int i=0; i<n; i++) 
	{
		v.x += c[i] * w[i].x;
		v.y += c[i] * w[i].y;
		v.z += c[i] * w[i].z;
	}
	return v;
}

Vector3D sum(int n, double *c, Vector3D *w) 
{  // vector sum
	Vector3D  v;

	for (int i=0; i<n; i++) 
	{
		v.x += c[i] * w[i].x;
		v.y += c[i] * w[i].y;
		v.z += c[i] * w[i].z;
	}
	return v;
}

}
