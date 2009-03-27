#ifndef _TORCHVISION_SCANNING_PROFILE_MAT_H_
#define _TORCHVISION_SCANNING_PROFILE_MAT_H_

namespace Torch
{
	//////////////////////////////////////////////////////////////////////////
	// Matrix operations - numerical recipes & JAMA package
	//////////////////////////////////////////////////////////////////////////

	// Invert matrixes and return its determinant
	double mat_invert(double** mat, double** inv_mat, int size);

	// Get the eigen vectors and eigen values of some symetrical matrix
	bool mat_eigen_sym(double** mat, double** evectors, double* evalues, int size);

	// Computes the square root of some symetrical matrix
	bool mat_sqrt_sym(double** mat, double** sqrt_mat, int size);
}

#endif
