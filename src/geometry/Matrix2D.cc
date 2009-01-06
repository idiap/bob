#include "Matrix2D.h"

namespace Torch {

Matrix2D::Matrix2D() : Mat(2, 2)
{
	ptr[0][0] = 0; ptr[0][1] = 0;
	ptr[1][0] = 0; ptr[1][1] = 0;
}

Matrix2D::Matrix2D(double x00, double x01, double x10, double x11) : Mat(2, 2)
{
	ptr[0][0] = x00; ptr[0][1] = x01;
	ptr[1][0] = x10; ptr[1][1] = x11;
}

Matrix2D::Matrix2D(const Matrix2D &w) : Mat(2, 2)
{
	for(int i = 0 ; i < m ; i++)
		for(int j = 0 ; j < n ; j++)
		   	ptr[i][j] = w.ptr[i][j];
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

	for(int i = 0 ; i < m ; i++)
		for(int j = 0 ; j < n ; j++)
		   	m_.ptr[i][j] = -ptr[i][j];

	return m_;
}

// equal
Matrix2D Matrix2D::operator=(Matrix2D w)
{
	for(int i = 0 ; i < m ; i++)
		for(int j = 0 ; j < n ; j++)
		   	ptr[i][j] = w.ptr[i][j];

	return *this;
}

//------------------------------------------------------------------
// IO streams
//------------------------------------------------------------------

// Write output
void Matrix2D::saveFile(File *file)
{
   	file->write(&m, sizeof(int), 1);
   	file->write(&n, sizeof(int), 1);

	for(int i = 0 ; i < m ; i++)
	{
		for(int j = 0 ; j < n ; j++)
		{
		   	double e = ptr[i][j];
			file->write(&e, sizeof(double), 1);
		}
	}
}

// Read input
void Matrix2D::loadFile(File *file)
{
   	file->read(&m, sizeof(int), 1);
   	file->read(&n, sizeof(int), 1);

	for(int i = 0 ; i < m ; i++)
	{
		for(int j = 0 ; j < n ; j++)
		{
		   	double e;
			file->read(&e, sizeof(double), 1);
			ptr[i][j] = e;
		}
	}
}

const char * Matrix2D::sprint()
{
   	switch(m)
	{
	case 2:
		sprintf(buf_sprint, "[{%g, %g} {%g, %g}]",
		      			ptr[0][0], ptr[0][1],
		      			ptr[1][0], ptr[1][1]);
		break;
	case 3:
		sprintf(buf_sprint, "[{%g, %g, %g} {%g, %g, %g} {%g, %g, %g}]",
		      			ptr[0][0], ptr[0][1], ptr[0][2],
		      			ptr[1][0], ptr[1][1], ptr[1][2],
		      			ptr[2][2], ptr[2][2], ptr[2][2]);
		break;
	}

	return buf_sprint;
}

//------------------------------------------------------------------
//  Scalar Ops
//------------------------------------------------------------------

// Scalar multiplication
Matrix2D operator*(int c, Matrix2D w)
{
	Matrix2D m_;
	for(int i = 0 ; i < w.m ; i++)
		for(int j = 0 ; j < w.n ; j++)
		   	m_.ptr[i][j] = c * w.ptr[i][j];
	return m_;
}

Matrix2D operator*(double c, Matrix2D w)
{
	Matrix2D m_;
	for(int i = 0 ; i < w.n ; i++)
		for(int j = 0 ; j < w.n ; j++)
		   	m_.ptr[i][j] = c * w.ptr[i][j];
	return m_;
}

Matrix2D operator*(Matrix2D w, int c)
{
	Matrix2D m_;
	for(int i = 0 ; i < w.m ; i++)
		for(int j = 0 ; j < w.n ; j++)
		   	m_.ptr[i][j] = c * w.ptr[i][j];
	return m_;
}

Matrix2D operator*(Matrix2D w, double c)
{
	Matrix2D m_;
	for(int i = 0 ; i < w.m ; i++)
		for(int j = 0 ; j < w.n ; j++)
		   	m_.ptr[i][j] = c * w.ptr[i][j];
	return m_;
}

// Scalar division
Matrix2D operator/(Matrix2D w, int c)
{
	Matrix2D m_;
	for(int i = 0 ; i < w.m ; i++)
		for(int j = 0 ; j < w.n ; j++)
		   	m_.ptr[i][j] = w.ptr[i][j] / c;
	return m_;
}

Matrix2D operator/(Matrix2D w, double c)
{
	Matrix2D m_;
	for(int i = 0 ; i < w.m ; i++)
		for(int j = 0 ; j < w.n ; j++)
		   	m_.ptr[i][j] = w.ptr[i][j] / c;
	return m_;
}

//------------------------------------------------------------------
//  Arithmetic Ops
//------------------------------------------------------------------

Matrix2D Matrix2D::operator+(Matrix2D w)
{
	Matrix2D m_;
	for(int i = 0 ; i < m ; i++)
		for(int j = 0 ; j < n ; j++)
		   	m_.ptr[i][j] = ptr[i][j] + w.ptr[i][j];
	return m_;
}

Matrix2D Matrix2D::operator-(Matrix2D w)
{
	Matrix2D m_;
	for(int i = 0 ; i < m ; i++)
		for(int j = 0 ; j < n ; j++)
		   	m_.ptr[i][j] = ptr[i][j] - w.ptr[i][j];
	return m_;
}

//------------------------------------------------------------------
//  Shorthand Ops
//------------------------------------------------------------------

Matrix2D& Matrix2D::operator*=(double c)
{        // matrix scalar mult
	for(int i = 0 ; i < m ; i++)
		for(int j = 0 ; j < n ; j++)
		   	ptr[i][j] *= c;
	return *this;
}

Matrix2D& Matrix2D::operator/=(double c)
{        // matrix scalar div
	for(int i = 0 ; i < m ; i++)
		for(int j = 0 ; j < n ; j++)
		   	ptr[i][j] /= c;
	return *this;
}

Matrix2D& Matrix2D::operator+=(Matrix2D w)
{        // matrix increment
	for(int i = 0 ; i < m ; i++)
		for(int j = 0 ; j < n ; j++)
		   	ptr[i][j] += w.ptr[i][j];
	return *this;
}

Matrix2D& Matrix2D::operator-=(Matrix2D w)
{        // matrix decrement
	for(int i = 0 ; i < m ; i++)
		for(int j = 0 ; j < n ; j++)
		   	ptr[i][j] -= w.ptr[i][j];
	return *this;
}

}
