#ifndef MATRIX2D_INC
#define MATRIX2D_INC

#include "general.h"
#include "File.h"
#include "matrix.h"

namespace Torch {

/** This class is designed to handle a 2D matrix for geometric operations

    @author Sebastien Marcel (marcel@idiap.ch)
    @author Dan Sunday (http://www.softsurfer.com/)
    @version 2.1
    \Date
    @since 1.0
*/
class Matrix2D : public Mat
{
	char buf_sprint[250];
public:

	/** @name constructors */
	//@{
	/// creates an zero 2D matrix
	Matrix2D();

	/** creates a 2D matrix from coefficients.

	    the 2D matrix will be:
	    \begin{verbatim}
		[ a b
		  c d ]
	    \end{verbatim}
	*/
	Matrix2D(double a, double b, double c, double d);

	/// creates a 2D matrix from another one
	Matrix2D(const Matrix2D &);
	//@}

	///
	~Matrix2D();

	//-----

	/// save the matrix
	void saveFile(File *file);

	/// load a matrix
	void loadFile(File *file);

	/// print the matrix into a string
	const char *sprint();

	//-----

	/// copy a matrix
	Matrix2D operator=(Matrix2D);

	//-----

	/// unary minus operation
	Matrix2D operator-();

	//-----

	/** @name Matrix2D arithmetic operations */
	//@{
	/// addition
	Matrix2D operator+(Matrix2D);

	/// substraction
	Matrix2D operator-(Matrix2D);

	/// scalar multiplication
	Matrix2D& operator*=(double);

	/// scalar division
	Matrix2D& operator/=(double);

	/// increment
	Matrix2D& operator+=(Matrix2D);

	/// decrement
	Matrix2D& operator-=(Matrix2D);
	//@}
};

//-----

/** @name scalar multiplication

    @author Sebastien Marcel (marcel@idiap.ch)
    @author Dan Sunday (http://www.softsurfer.com/)
    @version 2.1
    \Date
    @since 1.0
*/
//@{
/// Matrix2D operator*(int, Matrix2D)
Matrix2D operator*(int, Matrix2D);

/// Matrix2D operator*(double, Matrix2D)
Matrix2D operator*(double, Matrix2D);

/// Matrix2D operator*(Matrix2D, int)
Matrix2D operator*(Matrix2D, int);

/// Matrix2D operator*(Matrix2D, double)
Matrix2D operator*(Matrix2D, double);
//@}

//-----

/** @name scalar division

    @author Sebastien Marcel (marcel@idiap.ch)
    @author Dan Sunday (http://www.softsurfer.com/)
    @version 2.1
    \Date
    @since 1.0
*/
//@{
/// Matrix2D operator/(Matrix2D, int)
Matrix2D operator/(Matrix2D, int);

/// Matrix2D operator/(Matrix2D, double)
Matrix2D operator/(Matrix2D, double);
//@}

}

#endif
