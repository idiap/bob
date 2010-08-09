#ifndef MATRIX2D_INC
#define MATRIX2D_INC

#include "core/general.h"
#include "core/Tensor.h"

namespace Torch {

	/** This class is designed to handle a 2D matrix for geometric operations

	    @author Sebastien Marcel (marcel@idiap.ch)
	    @author Dan Sunday (http://www.softsurfer.com/)
	    @version 2.1
	    \date
	    @since 1.0
	*/
	class Matrix2D : public DoubleTensor
	{
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

		/// copy a matrix
		Matrix2D operator=(const Matrix2D&);

		//-----

		/// unary minus operation
		Matrix2D operator-();

		//-----

		/** @name Matrix2D arithmetic operations */
		//@{
		/// addition
		Matrix2D operator+(const Matrix2D&);

		/// substraction
		Matrix2D operator-(const Matrix2D&);

		/// scalar multiplication
		Matrix2D& operator*=(double);

		/// scalar division
		Matrix2D& operator/=(double);

		/// increment
		Matrix2D& operator+=(const Matrix2D&);

		/// decrement
		Matrix2D& operator-=(const Matrix2D&);
		//@}
	};

	//-----

	/** @name scalar multiplication

	    @author Sebastien Marcel (marcel@idiap.ch)
	    @author Dan Sunday (http://www.softsurfer.com/)
	    @version 2.1
	    \date
	    @since 1.0
	*/
	//@{
	/// Matrix2D operator*(int, Matrix2D)
	Matrix2D operator*(int, const Matrix2D&);

	/// Matrix2D operator*(double, Matrix2D)
	Matrix2D operator*(double, const Matrix2D&);

	/// Matrix2D operator*(Matrix2D, int)
	Matrix2D operator*(const Matrix2D&, int);

	/// Matrix2D operator*(Matrix2D, double)
	Matrix2D operator*(const Matrix2D&, double);
	//@}

	//-----

	/** @name scalar division

	    @author Sebastien Marcel (marcel@idiap.ch)
	    @author Dan Sunday (http://www.softsurfer.com/)
	    @version 2.1
	    \date
	    @since 1.0
	*/
	//@{
	/// Matrix2D operator/(Matrix2D, int)
	Matrix2D operator/(const Matrix2D&, int);

	/// Matrix2D operator/(Matrix2D, double)
	Matrix2D operator/(const Matrix2D&, double);
	//@}

}

#endif
