#ifndef MAT_INC
#define MAT_INC

#include "Vec.h"

namespace Torch {

/** Matrix object.

    @author Ronan Collobert (collober@idiap.ch)
*/
class Mat
{
public:
	/** Create a matrix from values in #ptr_#.
	    (No memory copy).
	*/
	Mat(double ** ptr_, int n_rows, int n_cols);

	/** Create a matrix from values in #ptr_#.
	    (No memory copy).
	*/
	Mat(double * ptr_, int n_rows, int n_cols);

	/// Create a new matrix
	Mat(int n_rows, int n_cols);

	/// Copy the matrix #mat#
	void copy(Mat * mat);

	/// Zero the matrix
	void zero();

	/// Compute the norm1
	double norm1();

	/// Compute the Frobenius norm
	double normFrobenius();

	/// Compute the norm inf
	double normInf();

	/** Return the row #row# of the matrix.
	    If #vec# is NULL, return a new vector.
	    Else copy the row in #vec#.
	*/
	Vec *getRow(int row, Vec * vec = 0);

	/** Return the column #col# of the matrix.
	    If #vec# is NULL, return a new vector.
	    Else copy the column in #vec#.
	*/
	Vec *getCol(int col, Vec * vec = 0);

	/// Set the row #row# to values in #vec#
	void setRow(int row, Vec * vec);

	/// Set the column #col# to values in #vec#
	void setCol(int row, Vec * vec);

	/** Return a sub-matrix.
	    Note that the memory is shared with the original
	    matrix, so *be carefull*.
	    You have to destroy the returned matrix.
	*/
	Mat *subMat(int row1, int col1, int row2, int col2);

	~Mat();

public:

	/// Size of the matrix
	int m, n;

	/// Data of the matrix
	double **ptr;

	/** NULL if not allocated by Mat.
	    (when you're using the first constructor of Mat, or for
	    the matrix returned by subMat)
	*/
	double *base;

	/// Tells if the <base> or <ptr> should be deallocated (shared or not)
	bool m_shouldDeleteBase;
	bool m_shouldDeletePtr;
};

}

#endif
