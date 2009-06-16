#include "MultiVariateDiagonalGaussianDistribution.h"

namespace Torch {

MultiVariateDiagonalGaussianDistribution::MultiVariateDiagonalGaussianDistribution()
{
	/*
	no_variances_update = no_variances_update_;
	use_log = use_log_;
	*/
}

MultiVariateDiagonalGaussianDistribution::MultiVariateDiagonalGaussianDistribution(int n_inputs_, int n_gaussians_) : MultiVariateNormalDistribution(n_inputs_, n_gaussians_)
{
}

MultiVariateDiagonalGaussianDistribution::~MultiVariateDiagonalGaussianDistribution()
{
}

bool MultiVariateDiagonalGaussianDistribution::prepare()
{
	return ProbabilityDistribution::prepare();
}

bool MultiVariateDiagonalGaussianDistribution::EMinit()
{
	return true;
}

void MultiVariateDiagonalGaussianDistribution::EMaccPosteriors(const DoubleTensor *input)
{
}

bool MultiVariateDiagonalGaussianDistribution::EMupdate()
{
	return true;
}
	
bool MultiVariateDiagonalGaussianDistribution::forward(const DoubleTensor *input)
{
	return true;
}

double MultiVariateDiagonalGaussianDistribution::sampleProbabilityOneGaussian(double *sample_, int g_)
{
	return true;
}

}

