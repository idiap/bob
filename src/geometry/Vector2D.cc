#include "Vector2D.h"
#include "Matrix2D.h"

namespace Torch {

//------------------------------------------------------------------
//  Unary Ops
//------------------------------------------------------------------

// Unary minus
Vector2D Vector2D::operator-() 
{
	Vector2D v;
	v.x = -x; v.y = -y;
	return v;
}

// Unary 2D perp operator
Vector2D Vector2D::operator~() 
{
	Vector2D v;
	v.x = -y; v.y = x;
	return v;
}

//------------------------------------------------------------------
//  Scalar Ops
//------------------------------------------------------------------

// Scalar multiplication
Vector2D operator*(int c, Vector2D w) 
{
	Vector2D v;
	v.x = c * w.x;
	v.y = c * w.y;
	return v;
}

Vector2D operator*(double c, Vector2D w) 
{
	Vector2D v;
	v.x = c * w.x;
	v.y = c * w.y;
	return v;
}

Vector2D operator*(Vector2D w, int c) 
{
	Vector2D v;
	v.x = c * w.x;
	v.y = c * w.y;
	return v;
}

Vector2D operator*(Vector2D w, double c) 
{
	Vector2D v;
	v.x = c * w.x;
	v.y = c * w.y;
	return v;
}

// Scalar division
Vector2D operator/(Vector2D w, int c) 
{
	Vector2D v;
	v.x = w.x / c;
	v.y = w.y / c;
	return v;
}

Vector2D operator/(Vector2D w, double c) 
{
	Vector2D v;
	v.x = w.x / c;
	v.y = w.y / c;
	return v;
}

//------------------------------------------------------------------
//  Arithmetic Ops
//------------------------------------------------------------------

Vector2D Vector2D::operator+(Vector2D w) 
{
	Vector2D v;
	v.x = x + w.x;
	v.y = y + w.y;
	return v;
}

Vector2D Vector2D::operator-(Vector2D w) 
{
	Vector2D v;
	v.x = x - w.x;
	v.y = y - w.y;
	return v;
}

//------------------------------------------------------------------
//  Products
//------------------------------------------------------------------

// Inner Dot Product
double Vector2D::operator*(Vector2D w) 
{
	return (x * w.x + y * w.y);
}

// 2D Exterior Perp Product
double Vector2D::operator|(Vector2D w) 
{
	return (x * w.y - y * w.x);
}

//------------------------------------------------------------------
//  Shorthand Ops
//------------------------------------------------------------------

Vector2D& Vector2D::operator*=(double c) 
{        // vector scalar mult
	x *= c;
	y *= c;
	return *this;
}

Vector2D& Vector2D::operator/=(double c) 
{        // vector scalar div
	x /= c;
	y /= c;
	return *this;
}

Vector2D& Vector2D::operator+=(Vector2D w) 
{        // vector increment
	x += w.x;
	y += w.y;
	return *this;
}

Vector2D& Vector2D::operator-=(Vector2D w) 
{        // vector decrement
	x -= w.x;
	y -= w.y;
	return *this;
}

Vector2D operator*(Vector2D w, Matrix2D m)
{
	Vector2D v;
	
	v.x = m.ptr[0][0] * w.x + m.ptr[1][0] * w.y;
	v.y = m.ptr[0][1] * w.x + m.ptr[1][1] * w.y;

	return v;
}

//------------------------------------------------------------------
//  Special Operations
//------------------------------------------------------------------

void Vector2D::normalize() 
{                      // convert to unit length
	double ln = sqrt(x*x + y*y);
	if (ln == 0) return;                    // do nothing for nothing
	x /= ln;
	y /= ln;
}

Vector2D sum(int n, int *c, Vector2D *w) 
{     // vector sum
	Vector2D  v;

	for (int i=0; i<n; i++) 
	{
		v.x += c[i] * w[i].x;
		v.y += c[i] * w[i].y;
	}
	return v;
}

Vector2D sum(int n, double *c, Vector2D *w) 
{  // vector sum
	Vector2D  v;

	for (int i=0; i<n; i++) 
	{
		v.x += c[i] * w[i].x;
		v.y += c[i] * w[i].y;
	}
	return v;
}

/** Get the angle (in radian) in the  range [-PI...PI] with the horizontal (x axes) 
*/
double Vector2D::angle()
{
	return atan2(y, x);
}

}
