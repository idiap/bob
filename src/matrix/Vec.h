#ifndef VEC_INC
#define VEC_INC

namespace Torch {

/** Vector object.

    @author Ronan Collobert (collober@idiap.ch)
*/
class Vec
{
public:
	/// Size of the vector
	int n;

	/// Data of the vector
	double *ptr;

	/** Create a vector from values in #ptr_#.
	    (No memory copy).
	*/
	Vec(double * ptr_, int n_dim);

	/// Create a new vector
	Vec(int n_dim);

	/// Copy the vector #vec# starting from index #start_i#
	void copy(Vec * vec, int start_i = 0);

	/// Zero the matrix
	void zero();

	/// Compute the norm1
	double norm1(Vec * weights = 0);

	/// Compute the norm2
	double norm2(Vec * weights = 0);

	/// Compute the norm inf
	double normInf();

	/// Inner product of two vectors from #start_i# downwards
	double iP(Vec * vec, int start_i = 0);

	/** Return a sub-vector.
	    Note that the memory is shared with the original
	    vector, so *be carefull*.
	*/
	Vec *subVec(int dim1, int dim2);

	~Vec();

	/// Tells is the <ptr> should be deallocated (shared or not)
	bool m_shouldDeletePtr;
};


}

#endif
