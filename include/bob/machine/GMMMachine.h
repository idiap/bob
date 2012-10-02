/**
 * @file bob/machine/GMMMachine.h
 * @date Tue May 10 11:35:58 2011 +0200
 * @author Francois Moulin <Francois.Moulin@idiap.ch>
 * @author Laurent El Shafey <Laurent.El-Shafey@idiap.ch>
 *
 * @brief This class implements a multivariate diagonal Gaussian distribution.
 * @details See Section 2.3.9 of Bishop, "Pattern recognition and machine learning", 2006
 *
 * Copyright (C) 2011-2012 Idiap Research Institute, Martigny, Switzerland
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

#ifndef BOB_MACHINE_GMMMACHINE_H
#define BOB_MACHINE_GMMMACHINE_H

#include "bob/machine/Machine.h"
#include "bob/machine/Gaussian.h"
#include "bob/machine/GMMStats.h"
#include "bob/io/HDF5File.h"
#include <iostream>
#include <boost/shared_ptr.hpp>
#include <vector>

namespace bob { namespace machine {
  
/** 
 * @brief This class implements a multivariate diagonal Gaussian distribution.
 * @details See Section 2.3.9 of Bishop, "Pattern recognition and machine learning", 2006
 */
class GMMMachine: public Machine<blitz::Array<double,1>, double> 
{
  public:
    /**
     * Default constructor
     */
    GMMMachine();

    /**
     * Constructor
     * @param[in] n_gaussians  The number of Gaussian components
     * @param[in] n_inputs     The feature dimensionality
     */
    GMMMachine(const size_t n_gaussians, const size_t n_inputs);

    /**
     * Copy constructor
     * (Needed because the GMM points to its constituent Gaussian members)
     */
    GMMMachine(const GMMMachine& other);

    /**
     * Constructor from a Configuration
     */
    GMMMachine(bob::io::HDF5File& config);

    /**
     * Assigment
     */
    GMMMachine& operator=(const GMMMachine &other);

    /**
     * Equal to
     */
    bool operator==(const GMMMachine& b) const;
    /**
     * Not equal to
     */
    bool operator!=(const GMMMachine& b) const;
    
    /**
     * Destructor
     */
    virtual ~GMMMachine(); 

    /**
     * Set the feature dimensionality
     */
    void setNInputs(const size_t n_inputs);

    /**
     * Get number of inputs
     */
    inline size_t getNInputs() const
    { return m_n_inputs; }

    /**
     * Reset the input dimensionality, and the number of Gaussian components.
     * Initialises the weights to uniform distribution.
     * @param n_gaussians The number of Gaussian components
     * @param n_inputs    The feature dimensionality
     */ 
    void resize(const size_t n_gaussians, const size_t n_inputs);

    /**
     * Set the weights
     */
    void setWeights(const blitz::Array<double,1> &weights);

    /**
     * Get the weights ("mixing coefficients") of the Gaussian components
     */
    inline const blitz::Array<double,1>& getWeights() const
    { return m_weights; }

    /**
     * Get the weights in order to be updated 
     * ("mixing coefficients") of the Gaussian components
     * @warning Only trainers should use this function for efficiency reason
     */
    inline blitz::Array<double,1>& updateWeights()
    { return m_weights; }

    /**
     * Get the logarithm of the weights of the Gaussian components
     */
    inline const blitz::Array<double,1>& getLogWeights() const
    { return m_cache_log_weights; }

    /**
     * Update the log of the weights in cache
     * @warning Should be used by trainer only when using updateWeights()
     */
    void recomputeLogWeights() const;

    /**
     * Set the means
     */
    void setMeans(const blitz::Array<double,2> &means);
    /**
     * Set the means from a supervector
     */
    void setMeanSupervector(const blitz::Array<double,1> &mean_supervector);
    /** 
     * Get the means
     */
    void getMeans(blitz::Array<double,2> &means) const;
    /**
     * Get the mean supervector
     */
    void getMeanSupervector(blitz::Array<double,1> &mean_supervector) const;
     /**
     * Returns a const reference to the supervector (Put in cache)
     */
    const blitz::Array<double,1>& getMeanSupervector() const;

    /**
     * Set the variances
     */
    void setVariances(const blitz::Array<double,2> &variances);
    /**
     * Set the variances from a supervector
     */
    void setVarianceSupervector(const blitz::Array<double,1> &variance_supervector);
    /**
     * Get the variances
     */
    void getVariances(blitz::Array<double,2> &variances) const;
    /**
     * Get the variance supervector
     */
    void getVarianceSupervector(blitz::Array<double,1> &variance_supervector) const;
    /**
     * Returns a const reference to the supervector (Put in cache)
     */
    const blitz::Array<double,1>& getVarianceSupervector() const;
    
    /**
     * Set the variance flooring thresholds in each dimension 
     */
    void setVarianceThresholds(const double value);
    /**
     * Set the variance flooring thresholds in each dimension
     * (equal for all Gaussian components)
     */
    void setVarianceThresholds(blitz::Array<double,1> variance_thresholds);
    /**
     * Set the variance flooring thresholds for each Gaussian in each dimension
     */
    void setVarianceThresholds(const blitz::Array<double,2> &variance_thresholds);
    /**
     * Get the variance flooring thresholds for each Gaussian in each dimension
     */
    void getVarianceThresholds(blitz::Array<double,2> &variance_thresholds) const;
    
    /**
     * Output the log likelihood of the sample, x, i.e. log(p(x|GMMMachine))
     * @param[in]  x                                 The sample
     * @param[out] log_weighted_gaussian_likelihoods For each Gaussian, i: log(weight_i*p(x|Gaussian_i))
     * @return     The GMMMachine log likelihood, i.e. log(p(x|GMMMachine))
     * Dimensions of the parameters are checked
     */
    double logLikelihood(const blitz::Array<double, 1> &x, blitz::Array<double,1> &log_weighted_gaussian_likelihoods) const;

    /**
     * Output the log likelihood of the sample, x, i.e. log(p(x|GMMMachine))
     * @param[in]  x                                 The sample
     * @param[out] log_weighted_gaussian_likelihoods For each Gaussian, i: log(weight_i*p(x|Gaussian_i))
     * @return     The GMMMachine log likelihood, i.e. log(p(x|GMMMachine))
     * @warning Dimensions of the parameters are not checked
     */
    double logLikelihood_(const blitz::Array<double, 1> &x, blitz::Array<double,1> &log_weighted_gaussian_likelihoods) const;

    /**
     * Output the log likelihood of the sample, x, i.e. log(p(x|GMM))
     * @param[in]  x The sample
     * Dimension of the input is checked
     */
    double logLikelihood(const blitz::Array<double, 1> &x) const;

    /**
     * Output the log likelihood of the sample, x, i.e. log(p(x|GMM))
     * @param[in]  x The sample
     * @warning Dimension of the input is not checked
     */
    double logLikelihood_(const blitz::Array<double, 1> &x) const;

    /**
     * Output the log likelihood of the sample, x 
     * (overrides Machine::forward)
     * Dimension of the input is checked
     */
    void forward(const blitz::Array<double,1>& input, double& output) const;
    
    /**
     * Output the log likelihood of the sample, x 
     * (overrides Machine::forward_)
     * @warning Dimension of the input is not checked
     */ 
    void forward_(const blitz::Array<double,1>& input, double& output) const;
    
    /**
     * Accumulates the GMM statistics over a set of samples.
     * @see bool accStatistics(const blitz::Array<double,1> &x, GMMStats stats)
     * Dimensions of the parameters are checked
     */
    void accStatistics(const blitz::Array<double,2>& input, GMMStats &stats) const;

    /**
     * Accumulates the GMM statistics over a set of samples.
     * @see bool accStatistics(const blitz::Array<double,1> &x, GMMStats stats)
     * @warning Dimensions of the parameters are not checked
     */
    void accStatistics_(const blitz::Array<double,2>& input, GMMStats &stats) const;

    /**
     * Accumulate the GMM statistics for this sample.
     *
     * @param[in]  x     The current sample
     * @param[out] stats The accumulated statistics
     * Dimensions of the parameters are checked
     */
    void accStatistics(const blitz::Array<double,1> &x, GMMStats &stats) const;

    /**
     * Accumulate the GMM statistics for this sample.
     *
     * @param[in]  x     The current sample
     * @param[out] stats The accumulated statistics
     * @warning Dimensions of the parameters are not checked
     */
    void accStatistics_(const blitz::Array<double,1> &x, GMMStats &stats) const;
    
    /**
     * Get a pointer to a particular Gaussian component
     * @param[in] i The index of the Gaussian component
     * @return A smart pointer to the i'th Gaussian component
     *         if it exists, otherwise throws an exception
     */
    boost::shared_ptr<bob::machine::Gaussian> getGaussian(const size_t i);

    /**
     * Return the number of Gaussian components
     */
    inline size_t getNGaussians() const
    { return m_n_gaussians; }

    /**
     * Save to a Configuration
     */
    void save(bob::io::HDF5File& config) const;
    
    /**
     * Load from a Configuration
     */
    void load(bob::io::HDF5File& config);

    /**
     * Load/Reload mean/variance supervector in cache
     */
    void reloadCacheSupervectors() const;
    
    friend std::ostream& operator<<(std::ostream& os, const GMMMachine& machine);

    
  private:
    /**
     * Copy another GMMMachine
     */
    void copy(const GMMMachine&);
    
    /**
     * The number of Gaussian components
     */
    size_t m_n_gaussians;
    
    /**
     * The feature dimensionality
     */
    size_t m_n_inputs;

    /**
     * The Gaussian components
     */
    std::vector<boost::shared_ptr<Gaussian> > m_gaussians;

    /**
     * The weights (also known as "mixing coefficients")
     */
    blitz::Array<double,1> m_weights;

    /**
     * Update the mean and variance supervectors 
     * in cache (into a 1D blitz array)
     */
    void updateCacheSupervectors() const;

    /**
     * Initialise the cache members (allocate arrays)
     */
    void initCache() const;

    /**
     * Accumulate the GMM statistics for this sample. 
     * Called by accStatistics() and accStatistics_()
     *
     * @param[in]  x     The current sample
     * @param[out] stats The accumulated statistics
     * @param[in]  log_likelihood  The current log_likelihood
     * @warning Dimensions of the parameters are not checked
     */
    void accStatisticsInternal(const blitz::Array<double,1> &x, 
      GMMStats &stats, const double log_likelihood) const;
    

    /// Some cache arrays to avoid re-allocation when computing log-likelihoods
    mutable blitz::Array<double,1> m_cache_log_weights;
    mutable blitz::Array<double,1> m_cache_log_weighted_gaussian_likelihoods;
    mutable blitz::Array<double,1> m_cache_P;
    mutable blitz::Array<double,2> m_cache_Px;

    mutable blitz::Array<double,1> m_cache_mean_supervector;
    mutable blitz::Array<double,1> m_cache_variance_supervector;
    mutable bool m_cache_supervector;
    
};

}}

#endif
