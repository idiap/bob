/**
 * @file cxx/ip/src/Matrix2D.cc
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
#include "ip/Matrix2D.h"

namespace bob {

Matrix2D::Matrix2D() : DoubleTensor(2, 2)
{
	fill(0.0);
}

Matrix2D::Matrix2D(double x00, double x01, double x10, double x11) : DoubleTensor(2, 2)
{
	set(0, 0, x00); set(0, 1, x01);
	set(1, 0, x10); set(1, 1, x11);
}

Matrix2D::Matrix2D(const Matrix2D &w) : DoubleTensor(2, 2)
{
	copy(&w);
}

Matrix2D::~Matrix2D()
{
}

//------------------------------------------------------------------
//  Unary Ops
//------------------------------------------------------------------

// Unary minus
Matrix2D Matrix2D::operator-()
{
   	Matrix2D m_;

	for(int i = 0 ; i < 2 ; i++)
		for(int j = 0 ; j < 2 ; j++)
		   	m_.set(i, j, -get(i, j));

	return m_;
}

// equal
Matrix2D Matrix2D::operator=(const Matrix2D& w)
{
	return Matrix2D(w);
}

//------------------------------------------------------------------
//  Scalar Ops
//------------------------------------------------------------------

// Scalar multiplication
Matrix2D operator*(int c, const Matrix2D& w)
{
	Matrix2D m_;
	for(int i = 0 ; i < 2 ; i++)
		for(int j = 0 ; j < 2 ; j++)
		   	m_.set(i, j, (double)c * w.get(i, j));
	return m_;
}

Matrix2D operator*(double c, const Matrix2D& w)
{
	Matrix2D m_;
	for(int i = 0 ; i < 2 ; i++)
		for(int j = 0 ; j < 2 ; j++)
		   	m_.set(i, j, c * w.get(i, j));
	return m_;
}

Matrix2D operator*(const Matrix2D& w, int c)
{
	Matrix2D m_;
	for(int i = 0 ; i < 2 ; i++)
		for(int j = 0 ; j < 2 ; j++)
		   	m_.set(i, j, (double)c * w.get(i, j));;
	return m_;
}

Matrix2D operator*(const Matrix2D& w, double c)
{
	Matrix2D m_;
	for(int i = 0 ; i < 2 ; i++)
		for(int j = 0 ; j < 2 ; j++)
		   	m_.set(i, j, c * w.get(i, j));
	return m_;
}

// Scalar division
Matrix2D operator/(const Matrix2D& w, int c)
{
	Matrix2D m_;
	for(int i = 0 ; i < 2 ; i++)
		for(int j = 0 ; j < 2 ; j++)
		   	m_.set(i, j, w.get(i, j) / c);
	return m_;
}

Matrix2D operator/(const Matrix2D& w, double c)
{
	Matrix2D m_;
	for(int i = 0 ; i < 2 ; i++)
		for(int j = 0 ; j < 2 ; j++)
		   	m_.set(i, j, w.get(i, j) / c);
	return m_;
}

//------------------------------------------------------------------
//  Arithmetic Ops
//------------------------------------------------------------------

Matrix2D Matrix2D::operator+(const Matrix2D& w)
{
	Matrix2D m_;
	for(int i = 0 ; i < 2 ; i++)
		for(int j = 0 ; j < 2 ; j++)
		   	m_.set(i, j, get(i, j) + w.get(i, j));
	return m_;
}

Matrix2D Matrix2D::operator-(const Matrix2D& w)
{
	Matrix2D m_;
	for(int i = 0 ; i < 2 ; i++)
		for(int j = 0 ; j < 2 ; j++)
		   	m_.set(i, j, get(i, j) - w.get(i, j));
	return m_;
}

//------------------------------------------------------------------
//  Shorthand Ops
//------------------------------------------------------------------

Matrix2D& Matrix2D::operator*=(double c)
{        // matrix scalar mult
	for(int i = 0 ; i < 2 ; i++)
		for(int j = 0 ; j < 2 ; j++)
		   	set(i, j, get(i, j) * c);
	return *this;
}

Matrix2D& Matrix2D::operator/=(double c)
{        // matrix scalar div
	for(int i = 0 ; i < 2 ; i++)
		for(int j = 0 ; j < 2 ; j++)
		   	set(i, j, get(i, j) / c);
	return *this;
}

Matrix2D& Matrix2D::operator+=(const Matrix2D& w)
{        // matrix increment
	for(int i = 0 ; i < 2 ; i++)
		for(int j = 0 ; j < 2 ; j++)
		   	set(i, j, get(i, j) + w.get(i, j));
	return *this;
}

Matrix2D& Matrix2D::operator-=(const Matrix2D& w)
{        // matrix decrement
	for(int i = 0 ; i < 2 ; i++)
		for(int j = 0 ; j < 2 ; j++)
		   	set(i, j, get(i, j) - w.get(i, j));
	return *this;
}

}
