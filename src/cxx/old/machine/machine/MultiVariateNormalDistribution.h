/**
 * @file cxx/old/machine/machine/MultiVariateNormalDistribution.h
 * @date Wed Apr 6 14:49:40 2011 +0200
 * @author Andre Anjos <andre.anjos@idiap.ch>
 *
 * Copyright (C) 2011 Idiap Reasearch Institute, Martigny, Switzerland
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, version 3 of the License.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */
#ifndef _TORCH5SPRO_MULTIVARIATE_NORMAL_DISTRIBUTION_MACHINE_H_
#define _TORCH5SPRO_MULTIVARIATE_NORMAL_DISTRIBUTION_MACHINE_H_

#include "machine/ProbabilityDistribution.h"
#include "core/DataSet.h"
#include <boost/scoped_array.hpp>

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
	virtual bool 		EMaccPosteriors(const DoubleTensor *input, const double input_posterior);

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
	virtual bool 		sampleEMaccPosteriors(const DoubleTensor& sample_, const double input_posterior) = 0;

	///
	virtual double 		sampleProbability(const DoubleTensor& sample_) = 0;

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
	boost::scoped_array<double>   threshold_variances;

	//
	int best_mean;

	//
	double current_likelihood;
	boost::scoped_array<double>   current_likelihood_one_mean;

	//
	double global_likelihood;

	//---
	
	double acc_posteriors_sum_weights;
	boost::scoped_array<double>   acc_posteriors_weights;
	boost::scoped_array<double>   buffer_acc_posteriors_means;
	boost::scoped_array<double*>  acc_posteriors_means;
	boost::scoped_array<double>   buffer_acc_posteriors_variances;
  boost::scoped_array<double*>  acc_posteriors_variances;

	DoubleTensor *frame_;
	DoubleTensor *sequence_;
};
	
}

#endif


