/**
 * @file cxx/old/machine/src/MultiVariateMAPDiagonalGaussianDistribution.cc
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
#include "machine/MultiVariateMAPDiagonalGaussianDistribution.h"
#include <cmath>

static const double M_LOG2PI=std::log(M_2_PI);

namespace Torch {

MultiVariateMAPDiagonalGaussianDistribution::MultiVariateMAPDiagonalGaussianDistribution(MultiVariateDiagonalGaussianDistribution *prior_)
{
   	CHECK_FATAL(prior_ != NULL); 

	addFOption("map factor", 0.5, "map factor");
	addBOption("variance adapt", false, "adapt variance");
	addBOption("weight adapt", false, "weight variance");

	prior = prior_;

	n_inputs = prior->m_parameters->getI("n_inputs");
	n_means = prior->m_parameters->getI("n_means");

	double *means_ = prior->m_parameters->getDarray("means");
	prior_means.reset(new double*[n_means]);
	double *p = means_;
	for(int j = 0 ; j < n_means ; j++)
	{
		prior_means[j] = p; 
		p += n_inputs;
	}

	m_parameters->copy(prior->m_parameters);

	MultiVariateNormalDistribution::resize(n_inputs, n_means);
	resize(n_inputs, n_means);
}

MultiVariateMAPDiagonalGaussianDistribution::~MultiVariateMAPDiagonalGaussianDistribution()
{
}

bool MultiVariateMAPDiagonalGaussianDistribution::prepare()
{
	MultiVariateDiagonalGaussianDistribution::prepare();
	
	map_factor = getFOption("map factor");
	
	if(use_log == false)
	{
		error("MultiVariateMAPDiagonalGaussianDistribution::prepare() You have to use the log mode !!");
		
		return false;
	}

	return true;
}

bool MultiVariateMAPDiagonalGaussianDistribution::sampleEMaccPosteriors(const DoubleTensor& sample_, const double input_posterior)
{
   	/* Computes the posterior for each gaussian

		\begin{equation}
			P(q_j | x_i) = \frac{P(q_j) \times p(x_i | q_j)}{\sum_{k=1}^K P(q_k) \times p(x_i | q_k)}
		\end{equation}
	*/

   	/* Computes and keeps the numerator and cumulates the denominator
	*/
   	double posterior_denominator;
	if(use_log)
	{
		posterior_denominator = sampleProbability(sample_);
		for(int j = 0 ; j < n_means ; j++) posterior_numerator[j] = log(weights[j]) + current_likelihood_one_mean[j];
	}
	else
	{
   		posterior_denominator = 0.0;

		for(int j = 0 ; j < n_means ; j++) 
		{
			posterior_numerator[j] = weights[j] * sampleProbabilityOneGaussian(sample_, j); 
			posterior_denominator += posterior_numerator[j];
		}
	}

	/* Accumulates weights, means and variances weighted by the posterior

		\begin{equation}
			posterior_j = P(q_j | x_i)
			            = posterior_numerator[j] / posterior_denominator

			posterior_numerator[j]  = P(q_j) \times p(x_i | q_j)
						= weights[j] * sampleProbabilityOneGaussian(data[i], j)

			posterior_denominator   = \sum_{j=1}^{n_gaussians} P(q_j) \times p(x_i | q_j)
						= \sum_{j=1}^{n_gaussians} posterior_numerator[j]


			acc_posteriors_weights[j] = \sum_{i=1}^{n_samples} P(q_j | x_i)
						  = \sum_{i=1}^{n_samples} posterior_j

			acc_posteriors_sum_weights = \sum_{j=1}^{n_gaussians} \sum_{i=1}^{n_samples} P(q_j | x_i)
						   = \sum_{j=1}^{n_gaussians} \sum_{i=1}^{n_samples} posterior_j

			acc_posteriors_means[j]	= \sum_{i=1}^{n_samples} P(q_j | x_i) \times x_i

			acc_posteriors_variances[j] = \sum_{i=1}^{n_samples} P(q_j | x_i) \times x_i^2

		\end{equation}
	*/
	for(int j = 0 ; j < n_means ; j++)
	{
	   	double posterior_j;
		if(use_log) posterior_j = exp(input_posterior + posterior_numerator[j] - posterior_denominator);
		else posterior_j = posterior_numerator[j] / posterior_denominator;

		acc_posteriors_weights[j] += posterior_j;
		acc_posteriors_sum_weights += posterior_j;

		for(int k = 0 ; k < n_inputs ; k++) 
		{
			double z = sample_(k);

			acc_posteriors_means[j][k] += posterior_j * z;

			acc_posteriors_variances[j][k] += posterior_j * z * z;
		}
	}

	return true;
}

bool MultiVariateMAPDiagonalGaussianDistribution::EMupdate()
{
	double precision = 10 * DBL_EPSILON;
	float min_weights = getFOption("min weights");

	bool variance_adapt = getBOption("variance adapt");
	bool weight_adapt = getBOption("weight adapt");

	if(variance_adapt)
		warning("MultiVariateMAPDiagonalGaussianDistribution::EMupdate() sorry variance adaptation not implemented yet");

	if(weight_adapt)
		warning("MultiVariateMAPDiagonalGaussianDistribution::EMupdate() sorry weight adaptation not implemented yet");

	for(int j = 0 ; j < n_means ; j++)
	{
	   	if(acc_posteriors_weights[j] <= (min_weights + precision))
		{
			// copy the means from the prior distribution
			for(int k = 0 ; k < n_inputs ; k++)
				means[j][k] = prior_means[j][k];
		}
		else
		{
			// update the means from the prior distribution
			for(int k = 0 ; k < n_inputs ; k++)
			{
				//means[j][k] = (map_factor * prior_means[j][k]) + ((1 - map_factor) * acc_posteriors_means[j][k] / acc_posteriors_weights[j]);
				means[j][k] = ((1 - map_factor) * prior_means[j][k]) + (map_factor * acc_posteriors_means[j][k] / acc_posteriors_weights[j]);
				if(variance_adapt)
				{
				}
			}
		}
	
		if(weight_adapt)
		{
		}
	}

	if(weight_adapt)
	{
	}

	/*
	p_weights_acc = weights_acc;
	for (int i=0;i<n_gaussians;i++,p_weights_acc++)
	{
		if (*p_weights_acc <= (prior_weights + epsilon)) 
			warning("Gaussian %d of GMM is not used in EM",i);
		else 
		{
			real* p_var_i = var[i];
			real* p_means_acc_i = means_acc[i];
			real* p_var_acc_i = var_acc[i];
			real* p_means_prior_i = prior_distribution->means[i];
			real* p_var_prior_i = prior_distribution->var[i];
			for (int j=0;j<n_inputs;j++) 
			{
				real means_ml = *p_means_acc_i++ / *p_weights_acc;
				real means_map = weight_on_prior * *p_means_prior_i + (1 - weight_on_prior) * means_ml;
				real var_ml = *p_var_acc_i++ / *p_weights_acc - means_map * means_map;
				real map_prior_2 = (means_map - *p_means_prior_i) * (means_map - *p_means_prior_i++);
				real map_ml_2 = (means_map - means_ml) * (means_map - means_ml);
				real var_map = weight_on_prior * (*p_var_prior_i++ + map_prior_2) + (1 - weight_on_prior) * (var_ml + map_ml_2);
				*p_var_i++ = var_map >= var_threshold[j] ? var_map : var_threshold[j];
			}
		}
	}
	
	// then the weights
	real sum_weights_acc = 0;
	p_weights_acc = weights_acc;
	for (int i=0;i<n_gaussians;i++)
		sum_weights_acc += *p_weights_acc++;
	real *p_log_weights = log_weights;
	real *prior_log_weights = prior_distribution->log_weights;
	real log_sum = log(sum_weights_acc);
	p_weights_acc = weights_acc;
	for (int i=0;i<n_gaussians;i++)
		*p_log_weights++ = log(weight_on_prior * exp( *prior_log_weights++) + (1-weight_on_prior) * exp(log(*p_weights_acc++) - log_sum));
	*/

	if(use_log)
	{
		// pre-compute constants
		double cst_ = n_inputs * M_LOG2PI;
		for(int j = 0 ; j < n_means ; j++) 
		{
			double log_det = 0.0;
			for(int k = 0 ; k < n_inputs ; k++) log_det += log(variances[j][k]);
			g_norm[j] = cst_ + log_det;
		}
	}

	return true;
}
	
}

