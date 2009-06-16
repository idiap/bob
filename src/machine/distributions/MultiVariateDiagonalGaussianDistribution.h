#ifndef _TORCH5SPRO_MULTIVARIATE_DIAGONAL_GAUSSIAN_DISTRIBUTION_MACHINE_H_
#define _TORCH5SPRO_MULTIVARIATE_DIAGONAL_GAUSSIAN_DISTRIBUTION_MACHINE_H_

#include "MultiVariateNormalDistribution.h"
#include "Machines.h"

namespace Torch {

/** MultiVariateNormalDistribution

	@author Sebastien Marcel (marcel@idiap.ch)
*/
class MultiVariateDiagonalGaussianDistribution : public MultiVariateNormalDistribution
{
	double *posterior_numerator;
	double *g_norm;

public:	
	//
	MultiVariateDiagonalGaussianDistribution();

	//
	MultiVariateDiagonalGaussianDistribution(int n_inputs_, int n_gaussians_);

	//
	~MultiVariateDiagonalGaussianDistribution();

	//---
	
	///
	virtual bool 		prepare();

	///
	virtual bool 		EMinit();

	//
	virtual void 		EMaccPosteriors(const DoubleTensor *input);

	///
	virtual bool 		EMupdate();
		
	///
	virtual bool 		forward(const DoubleTensor *input);

	//
	virtual double 		sampleProbabilityOneGaussian(double *sample_, int g_);

};
	
}

#endif


