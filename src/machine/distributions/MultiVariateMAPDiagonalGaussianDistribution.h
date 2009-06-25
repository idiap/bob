#ifndef _TORCH5SPRO_MULTIVARIATE_MAP_DIAGONAL_GAUSSIAN_DISTRIBUTION_MACHINE_H_
#define _TORCH5SPRO_MULTIVARIATE_MAP_DIAGONAL_GAUSSIAN_DISTRIBUTION_MACHINE_H_

#include "MultiVariateDiagonalGaussianDistribution.h"
#include "Machines.h"

namespace Torch {

/** MultiVariateMAPDiagonalGaussianDistribution

	@author Sebastien Marcel (marcel@idiap.ch)
*/
class MultiVariateMAPDiagonalGaussianDistribution : public MultiVariateDiagonalGaussianDistribution
{
public:	
	///
	MultiVariateMAPDiagonalGaussianDistribution(MultiVariateDiagonalGaussianDistribution *prior_);

	///
	~MultiVariateMAPDiagonalGaussianDistribution();

	///
	virtual bool 		prepare();

	///
	virtual bool 		EMaccPosteriors(const DoubleTensor *input, const double input_posterior);

	///
	virtual bool 		EMupdate();
		
protected:
	MultiVariateDiagonalGaussianDistribution *prior;
	double **prior_means;

	float map_factor;
};
	
}

#endif


