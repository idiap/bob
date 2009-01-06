#ifndef POINT3D_INC
#define POINT3D_INC

#include "Point2D.h"

namespace Torch {

class Vector3D;

/** This class is designed to handle a point in 3D

    @author Sebastien Marcel (marcel@idiap.ch)
    @author Dan Sunday (http://www.softsurfer.com/)
    @version 2.1
    \Date
    @since 1.0
*/
class Point3D : public Point2D
{
public:
	/// z coordinate
	double z;

	//-----

	/** @name Lots of Constructors */
	//@{
	///
	Point3D() { z=0; }

	///
	Point3D(int a) : Point2D(a) { z=0; }

	///
	Point3D(double a) : Point2D(a) { z=0; }

	///
	Point3D(int a, int b) : Point2D(a, b) { z=0; }

	///
	Point3D(double a, double b) : Point2D(a, b) { z=0; }

	///
	Point3D(int a, int b, int c) : Point2D(a, b) { z=c; }

	///
	Point3D(double a, double b, double c) : Point2D(a, b) { z=c; }
	//@}

	/// destructor
	virtual ~Point3D() {};

	//-----

	/** @name reset functions */
	//@{
	///
	void reset(int a, int b, int c) { x=a; y=b; z=c; }

	///
	void reset(double a, double b, double c) { x=a; y=b; z=c; }
	//@}

	//-----

	/// save the 3D point into a file
	virtual void saveFile(File *file);

	/// load a 3D point from a file
	virtual void loadFile(File *file);

	/// return the 3D point into a string
	virtual const char *sprint();

	//-----

	/// draw a 3D point onto an image
	virtual void draw(Image *image, Color color);

	//-----

	/// equality
	int operator==(Point3D);

	/// difference
	int operator!=(Point3D);

	//---

	/** @name Point3D and Vector3D operations

	    These operations are not always valid. */
	//@{
	/// Vector difference
	Vector3D operator-(Point3D);

	/// Vector translate +
	Point3D  operator+(Vector3D);

	/// Vector translate -
	Point3D  operator-(Vector3D);

	/// Vector inc translate
	Point3D& operator+=(Vector3D);

	/// Vector dec translate
	Point3D& operator-=(Vector3D);
	//@}
};

//-----

/** @name Point3D scalar multiplication and divisions

    These Point3D scalar operations (convenient but often illegal)
    using any type of scalar (int, float, or double)
    are not valid for points in general,
    unless they are 'affine' as coeffs of
    a sum in which all the coeffs add to 1,
    such as: the sum (a*P + b*Q) with (a+b == 1).

    The programmer must enforce this (if they want to).

    @author Dan Sunday (http://www.softsurfer.com/)
    @version 2.1
    \Date
    @since 1.0
*/
//@{
/// Point3D operator*(int, Point3D)
Point3D operator*(int, Point3D);

/// Point3D operator*(double, Point3D)
Point3D operator*(double, Point3D);

/// Point3D operator*(Point3D, int)
Point3D operator*(Point3D, int);

/// Point3D operator*(Point3D, double)
Point3D operator*(Point3D, double);

/// Point3D operator/(Point3D, int)
Point3D operator/(Point3D, int);

/// Point3D operator/(Point3D, double)
Point3D operator/(Point3D, double);
//@}

//-----

/** @name Point3D addition

    Point3D addition (also convenient but often illegal)
    is not valid unless part of an affine sum.

    The programmer must enforce this (if they want to).

    @author Dan Sunday (http://www.softsurfer.com/)
    @version 2.1
    \Date
    @since 1.0
*/
//@{
/// Point3D operator+(Point3D, Point3D) : Adds two 3D points
Point3D operator+(Point3D, Point3D);
//@}

//-----

/** @name Point3D affine sums

    Check if coefficents sum to 1 and compute the weighted sum (even when not affine).

    @author Dan Sunday (http://www.softsurfer.com/)
    @version 2.0
    \Date
    @since 1.0
*/
//@{
/// affine sum from int
Point3D asum(int, int *, Point3D *);

/// affine sum from double
Point3D asum(int, double *, Point3D *);
//@}

//-----

/** @name Point3D distances

    @author Dan Sunday (http://www.softsurfer.com/)
    @version 2.0
    \Date
    @since 1.0
*/
//@{
/// Euclidean distance
double d(Point3D, Point3D);

/// Squared Euclidean distance.
double d2(Point3D, Point3D);
//@}

}

#endif
