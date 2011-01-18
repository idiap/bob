#ifndef _TORCH5SPRO_MULTIVARIATE_MAP_DIAGONAL_GAUSSIAN_DISTRIBUTION_MACHINE_H_
#define _TORCH5SPRO_MULTIVARIATE_MAP_DIAGONAL_GAUSSIAN_DISTRIBUTION_MACHINE_H_

#include "machine/MultiVariateDiagonalGaussianDistribution.h"
#include "machine/Machines.h"
#include <boost/scoped_array.hpp>

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
	virtual bool 		sampleEMaccPosteriors(const DoubleTensor& sample_, const double input_posterior);

	///
	virtual bool 		EMupdate();
		
protected:
	MultiVariateDiagonalGaussianDistribution *prior;
	boost::scoped_array<double*> prior_means;

	float map_factor;
};
	
}

#endif


