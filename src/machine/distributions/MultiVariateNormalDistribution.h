#ifndef _TORCH5SPRO_MULTIVARIATE_NORMAL_DISTRIBUTION_MACHINE_H_
#define _TORCH5SPRO_MULTIVARIATE_NORMAL_DISTRIBUTION_MACHINE_H_

#include "ProbabilityDistribution.h"

namespace Torch {

/** MultiVariateNormalDistribution

	@author Sebastien Marcel (marcel@idiap.ch)
*/
class MultiVariateNormalDistribution : public ProbabilityDistribution
{
public:	
	//
	MultiVariateNormalDistribution();

	//
	MultiVariateNormalDistribution(int n_inputs_, int n_gaussians_);

	//
	~MultiVariateNormalDistribution();

	//---
	
	///
	virtual bool 		shuffle();

	///
	virtual bool 		EMinit() = 0;

	//
	virtual void 		EMaccPosteriors(const DoubleTensor *input) = 0;

	///
	virtual bool 		EMupdate() = 0;
		
	///
	virtual bool 		forward(const DoubleTensor *input);

	//
	virtual double sampleProbabilityOneGaussian(double *sample_, int g) = 0;
	virtual double sampleProbability(double *sample_) = 0;

	/*
	virtual void initMeans(double **means_);
	virtual void initMeans(int n_data_, real **data_);
	virtual void initVariances(int n_data_, real **data_, real factor_variance_threshold_ = 0.1);
	*/

	///
	void 		print();
		
public:
	//
	int n_gaussians;

	//
	double *weights;
	double **means;
	double **variances;
	
	//
	double *threshold_variances;
	
	//
	int best_gaussian;

	//
	double current_likelihood;
	double *current_likelihood_one_gaussian;

	//
	double global_likelihood;

	//---
	
protected:
	double acc_posteriors_sum_weights;
	double *acc_posteriors_weights;
	double *buffer_acc_posteriors_means;
	double **acc_posteriors_means;
	double *buffer_acc_posteriors_variances;
	double **acc_posteriors_variances;

};
	
}

#endif


