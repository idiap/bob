#ifndef _TORCH5SPRO_MULTIVARIATE_NORMAL_DISTRIBUTION_MACHINE_H_
#define _TORCH5SPRO_MULTIVARIATE_NORMAL_DISTRIBUTION_MACHINE_H_

#include "ProbabilityDistribution.h"
#include "DataSet.h"

namespace Torch {

/** MultiVariateNormalDistribution

	@author Sebastien Marcel (marcel@idiap.ch)
*/
class MultiVariateNormalDistribution : public ProbabilityDistribution
{
public:	
	///
	MultiVariateNormalDistribution();

	///
	MultiVariateNormalDistribution(int n_inputs_, int n_means_);

	///
	~MultiVariateNormalDistribution();

	///
	virtual bool 		EMinit();

	///
	virtual bool 		forward(const DoubleTensor *input);

	///
	virtual bool 		print();
	
	//---

	///
	virtual bool 		resize(int n_inputs_, int n_means_);

	///
	virtual bool 		cleanup();

	///
	virtual double 		sampleProbability(double *sample_) = 0;

	///
	int			getNmeans() { return n_means; };
	double **		getMeans() { return means; };
	double **		getVariances() { return variances; };

	///
	virtual bool		setMeans(double **means_);

	///
	virtual bool		setMeans(DataSet *dataset_);

	///
	virtual bool		setVariances(double **variances_);

	///
	virtual bool		setVariances(double *stdv_, double factor_variance_threshold_ = 0.1);

	///
	virtual bool		setVarianceFlooring(double *stdv_, double factor_variance_threshold_ = 0.1);

	///
	virtual bool 		shuffle();

protected:
	//
	int n_means;
	double **means;
	double *weights;
	double **variances;

	//
	double *threshold_variances;

	//
	int best_mean;

	//
	double current_likelihood;
	double *current_likelihood_one_mean;

	//
	double global_likelihood;

	//---
	
	double acc_posteriors_sum_weights;
	double *acc_posteriors_weights;
	double *buffer_acc_posteriors_means;
	double **acc_posteriors_means;
	double *buffer_acc_posteriors_variances;
	double **acc_posteriors_variances;

	DoubleTensor *frame_;
	DoubleTensor *sequence_;
};
	
}

#endif


