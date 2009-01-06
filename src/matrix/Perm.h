#ifndef PERM_INC
#define PERM_INC

#include "Perm.h"

namespace Torch {

/** Permutation object.

    @author Ronan Collobert (collober@idiap.ch)
*/
class Perm
{
public:
	/// Size of the permutation
	int n;

	/// Data of the permutation
	int *ptr;

	/// Create a new permutation
	Perm(int n_dim);

	/// Copy the permutation #perm#
	void copy(Perm * perm);

	~Perm();
};


}

#endif
