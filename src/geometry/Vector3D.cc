#include "Vector3D.h"

namespace Torch {

//------------------------------------------------------------------
//  Unary Ops
//------------------------------------------------------------------

// Unary minus
Vector3D Vector3D::operator-()
{
	return Vector3D(-get(0), -get(1), -get(2));
}

// Unary 2D perp operator
Vector3D Vector3D::operator~()
{
	return Vector3D(-get(1), get(0), get(2));
}

//------------------------------------------------------------------
//  Scalar Ops
//------------------------------------------------------------------

// Scalar multiplication
Vector3D operator*(int c, const Vector3D& w)
{
	return	Vector3D(	w.get(0) * c,
				w.get(1) * c,
				w.get(2) * c);
}

Vector3D operator*(double c, const Vector3D& w)
{
	return	Vector3D(	w.get(0) * c,
				w.get(1) * c,
				w.get(2) * c);
}

Vector3D operator*(const Vector3D& w, int c)
{
	return	Vector3D(	w.get(0) * c,
				w.get(1) * c,
				w.get(2) * c);
}

Vector3D operator*(const Vector3D& w, double c)
{
	return	Vector3D(	w.get(0) * c,
				w.get(1) * c,
				w.get(2) * c);
}

// Scalar division
Vector3D operator/(const Vector3D& w, int c)
{
	return	Vector3D(	w.get(0) / c,
				w.get(1) / c,
				w.get(2) / c);
}

Vector3D operator/(const Vector3D& w, double c)
{
	return	Vector3D(	w.get(0) * c,
				w.get(1) * c,
				w.get(2) * c);
}

//------------------------------------------------------------------
//  Arithmetic Ops
//------------------------------------------------------------------

Vector3D Vector3D::operator+(const Vector3D& w)
{
	return	Vector3D(	get(0) + w.get(0),
				get(1) + w.get(1),
				get(2) + w.get(2));
}

Vector3D Vector3D::operator-(const Vector3D& w)
{
	return	Vector3D(	get(0) - w.get(0),
				get(1) - w.get(1),
				get(2) - w.get(2));
}

//------------------------------------------------------------------
//  Products
//------------------------------------------------------------------

// Inner Dot Product
double Vector3D::operator*(const Vector3D& w)
{
	return (get(0) * w.get(0) + get(1) * w.get(1) + get(2) * w.get(2));
}

// 2D Exterior Perp Product
double Vector3D::operator|(const Vector3D& w)
{
	return (get(0) * w.get(1) - get(1) * w.get(0));
}

// 3D Exterior Cross Product
Vector3D Vector3D::operator^(const Vector3D& w)
{
	return Vector3D(get(1) * w.get(2) - get(2) * w.get(1),
			get(2) * w.get(0) - get(0) * w.get(2),
			get(0) * w.get(1) - get(1) * w.get(0));
}

//------------------------------------------------------------------
//  Shorthand Ops
//------------------------------------------------------------------

Vector3D& Vector3D::operator*=(double c)
{        // vector scalar mult
	set(0, get(0) * c);
	set(1, get(1) * c);
	set(2, get(2) * c);
	return *this;
}

Vector3D& Vector3D::operator/=(double c)
{        // vector scalar div
	set(0, get(0) / c);
	set(1, get(1) / c);
	set(2, get(2) / c);
	return *this;
}

Vector3D& Vector3D::operator+=(const Vector3D& w)
{        // vector increment
	set(0, get(0) + w.get(0));
	set(1, get(1) + w.get(1));
	set(2, get(2) + w.get(2));
	return *this;
}

Vector3D& Vector3D::operator-=(const Vector3D& w)
{        // vector decrement
	set(0, get(0) - w.get(0));
	set(1, get(1) - w.get(1));
	set(2, get(2) - w.get(2));
	return *this;
}

Vector3D& Vector3D::operator^=(const Vector3D& w)
{        // 3D exterior cross product
	double ox=get(0), oy=get(1), oz=get(2);
	set(0, oy * w.get(2) - oz * w.get(1));
	set(1, oz * w.get(0) - ox * w.get(2));
	set(2, ox * w.get(1) - oy * w.get(0));
	return *this;
}

//------------------------------------------------------------------
//  Special Operations
//------------------------------------------------------------------

void Vector3D::normalize()
{                      // convert to unit length
	double ln = len();
	if (ln == 0) return;                    // do nothing for nothing
	operator/=(ln);
}

Vector3D sum(int n, int *c, const Vector3D *w)
{     // vector sum
	Vector3D  v;

	for (int i=0; i<n; i++)
	{
		v.set(0, v.get(0) + c[i] * w[i].get(0));
		v.set(1, v.get(1) + c[i] * w[i].get(1));
		v.set(2, v.get(2) + c[i] * w[i].get(2));
	}
	return v;
}

Vector3D sum(int n, double *c, const Vector3D *w)
{  // vector sum
	Vector3D  v;

	for (int i=0; i<n; i++)
	{
		v.set(0, v.get(0) + c[i] * w[i].get(0));
		v.set(1, v.get(1) + c[i] * w[i].get(1));
		v.set(2, v.get(2) + c[i] * w[i].get(2));
	}
	return v;
}

}
