/**
 * @file cxx/ip/src/Vector2D.cc
 * @date Tue Jan 18 17:07:26 2011 +0100
 * @author André Anjos <andre.anjos@idiap.ch>
 *
 * Copyright (C) 2011 Idiap Reasearch Institute, Martigny, Switzerland
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, version 3 of the License.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */
#include "ip/Vector2D.h"
#include "ip/Matrix2D.h"

namespace Torch {

//------------------------------------------------------------------
// vector length

double Vector2D::len()
{
	return sqrt(get(0) * get(0) + get(1) * Point2D::get(1));
}

double Vector2D::len2()
{
	return get(0) * get(0) + get(1) * get(1);
}

//------------------------------------------------------------------
//  Unary Ops
//------------------------------------------------------------------

// Unary minus
Vector2D Vector2D::operator-()
{
	return Vector2D(-get(0), -get(1));
}

// Unary 2D perp operator
Vector2D Vector2D::operator~()
{
	return Vector2D(-get(1), get(0));
}

//------------------------------------------------------------------
//  Scalar Ops
//------------------------------------------------------------------

// Scalar multiplication
Vector2D operator*(int c, const Vector2D& w)
{
	return Vector2D(c * w.get(0), c * w.get(1));
}

Vector2D operator*(double c, const Vector2D& w)
{
	return Vector2D(c * w.get(0), c * w.get(1));
}

Vector2D operator*(const Vector2D& w, int c)
{
	return Vector2D(c * w.get(0), c * w.get(1));
}

Vector2D operator*(const Vector2D& w, double c)
{
	return Vector2D(c * w.get(0), c * w.get(1));
}

// Scalar division
Vector2D operator/(const Vector2D& w, int c)
{
	return Vector2D(w.get(0) / c, w.get(1) / c);
}

Vector2D operator/(const Vector2D& w, double c)
{
	return Vector2D(w.get(0) / c, w.get(1) / c);
}

//------------------------------------------------------------------
//  Arithmetic Ops
//------------------------------------------------------------------

Vector2D Vector2D::operator+(const Vector2D& w)
{
	return Vector2D(get(0) + w.get(0),
			get(1) + w.get(1));
}

Vector2D Vector2D::operator-(const Vector2D& w)
{
	return Vector2D(get(0) - w.get(0),
			get(1) - w.get(1));
}

//------------------------------------------------------------------
//  Products
//------------------------------------------------------------------

// Inner Dot Product
double Vector2D::operator*(const Vector2D& w)
{
	return get(0) * w.get(0) + get(1) * w.get(1);
}

// 2D Exterior Perp Product
double Vector2D::operator|(const Vector2D& w)
{
	return get(0) * w.get(1) - get(1) * w.get(0);
}

//------------------------------------------------------------------
//  Shorthand Ops
//------------------------------------------------------------------

Vector2D& Vector2D::operator*=(double c)
{        // vector scalar mult
	set(0, get(0) * c);
	set(1, get(1) * c);
	return *this;
}

Vector2D& Vector2D::operator/=(double c)
{        // vector scalar div
	set(0, get(0) / c);
	set(1, get(1) / c);
	return *this;
}

Vector2D& Vector2D::operator+=(const Vector2D& w)
{        // vector increment
	set(0, get(0) + w.get(0));
	set(1, get(1) + w.get(1));
	return *this;
}

Vector2D& Vector2D::operator-=(const Vector2D& w)
{        // vector decrement
	set(0, get(0) - w.get(0));
	set(1, get(1) - w.get(1));
	return *this;
}

Vector2D operator*(const Vector2D& w, const Matrix2D& m)
{
	return Vector2D(	m.get(0, 0) * w.get(0) + m.get(1, 0) * w.get(1),
				m.get(0, 1) * w.get(0) + m.get(1, 1) * w.get(1));
}

//------------------------------------------------------------------
//  Special Operations
//------------------------------------------------------------------

void Vector2D::normalize()
{                      // convert to unit length
	double ln = len();
	if (ln == 0) return;                    // do nothing for nothing
	operator/=(ln);
}

Vector2D sum(int n, int *c, const Vector2D *w)
{     // vector sum
	Vector2D  v;

	for (int i=0; i<n; i++)
	{
		v.set(0, v.get(0) + c[i] * w[i].get(0));
		v.set(1, v.get(1) + c[i] * w[i].get(1));
	}
	return v;
}

Vector2D sum(int n, double *c, const Vector2D *w)
{  // vector sum
	Vector2D  v;

	for (int i=0; i<n; i++)
	{
		v.set(0, v.get(0) + c[i] * w[i].get(0));
		v.set(1, v.get(1) + c[i] * w[i].get(1));
	}
	return v;
}

/** Get the angle (in radian) in the  range [-PI...PI] with the horizontal (x axes)
*/
double Vector2D::angle()
{
	return atan2(get(1), get(0));
}

}
