/**
 * @file cxx/machine/machine/GMMMachine.h
 * @date Tue May 10 11:35:58 2011 +0200
 * @author Francois Moulin <Francois.Moulin@idiap.ch>
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
/// @file GMMMachine.h
/// @author <a href="mailto:Roy.Wallace@idiap.ch">Roy Wallace</a> 
/// @author <a href="mailto:Laurent.El-Shafey@idiap.ch">Laurent El Shafey</a> 
/// @brief This class implements a multivariate diagonal Gaussian distribution.
/// @details See Section 2.3.9 of Bishop, "Pattern recognition and machine learning", 2006

#ifndef TORCH5SPRO_MACHINE_GMMMACHINE_H
#define TORCH5SPRO_MACHINE_GMMMACHINE_H

#include "io/Arrayset.h"
#include "machine/Machine.h"
#include "machine/GMMStats.h"
#include "machine/Gaussian.h"
#include "io/HDF5File.h"
#include <iostream>

namespace Torch {
namespace machine {
  
/// @brief This class implements a multivariate diagonal Gaussian distribution.
/// @details See Section 2.3.9 of Bishop, "Pattern recognition and machine learning", 2006
class GMMMachine: public Machine<blitz::Array<double,1>, double> {
  public:

    /// Default constructor
    GMMMachine();

    /// Constructor
    /// @param[in] n_gaussians  The number of Gaussian components
    /// @param[in] n_inputs     The feature dimensionality
    GMMMachine(int n_gaussians, int n_inputs);

    /// Constructor from a Configuration
    GMMMachine(Torch::io::HDF5File& config);

    /// Copy constructor
    /// (Needed because the GMM points to its constituent Gaussian members)
    GMMMachine(const GMMMachine& other);

    /// Assigment
    GMMMachine& operator=(const GMMMachine &other);

    /// Equal to
    bool operator==(const GMMMachine& b) const;
    
    /// Destructor
    virtual ~GMMMachine(); 

    /// Set the feature dimensionality
    /// Overrides Machine::setNInputs
    void setNInputs(int n_inputs);

    /// Get number of inputs
    int getNInputs() const;

    /// Reset the input dimensionality, and the number of Gaussian components.
    /// Initialises the weights to uniform distribution.
    /// @param n_gaussians The number of Gaussian components
    /// @param n_inputs    The feature dimensionality
    void resize(int n_gaussians, int n_inputs);

    /// Set the weights
    void setWeights(const blitz::Array<double,1> &weights);

    /// Set the means
    void setMeans(const blitz::Array<double,2> &means);
    /// Set the means from a supervector
    void setMeanSupervector(const blitz::Array<double,1> &mean_supervector);

    /// Get the means
    void getMeans(blitz::Array<double,2> &means) const;
    /// Get the mean supervector
    void getMeanSupervector(blitz::Array<double,1> &mean_supervector) const;
    
    /// Set the variances
    void setVariances(const blitz::Array<double,2> &variances);
    /// Set the variances from a supervector
    void setVarianceSupervector(const blitz::Array<double,1> &variance_supervector);
    /// Returns a const reference to the supervector (Put in cache)
    const blitz::Array<double,1>& getMeanSupervector() const;

    /// Get the variances
    void getVariances(blitz::Array<double,2> &variances) const;
    /// Get the variance supervector
    void getVarianceSupervector(blitz::Array<double,1> &variance_supervector) const;
    /// Returns a const reference to the supervector (Put in cache)
    const blitz::Array<double,1>& getVarianceSupervector() const;
    
    /// Set the variance flooring thresholds in each dimension 
    /// to a proportion of the current variance, for each Gaussian
    void setVarianceThresholds(double factor);

    /// Set the variance flooring thresholds in each dimension
    /// (equal for all Gaussian components)
    void setVarianceThresholds(blitz::Array<double,1> variance_thresholds);

    /// Set the variance flooring thresholds for each Gaussian in each dimension
    void setVarianceThresholds(const blitz::Array<double,2> &variance_thresholds);

    /// Get the variance flooring thresholds for each Gaussian in each dimension
    void getVarianceThresholds(blitz::Array<double,2> &variance_thresholds) const;
    
    /// Output the log likelihood of the sample, x, i.e. log(p(x|GMMMachine))
    /// @param[in]  x                                 The sample
    /// @param[out] log_weighted_gaussian_likelihoods For each Gaussian, i: log(weight_i*p(x|Gaussian_i))
    /// @return     The GMMMachine log likelihood, i.e. log(p(x|GMMMachine))
    double logLikelihood(const blitz::Array<double, 1> &x, blitz::Array<double,1> &log_weighted_gaussian_likelihoods) const;

    /// Output the log likelihood of the sample, x, i.e. log(p(x|GMM))
    /// @param[in]  x The sample
    double logLikelihood(const blitz::Array<double, 1> &x) const;

    /// Output the log likelihood of the sample, x 
    /// (overrides Machine::forward)
    /// Dimension of the input is checked
    void forward(const blitz::Array<double,1>& input, double& output) const;
    
    /// Output the log likelihood of the sample, x 
    /// (overrides Machine::forward_)
    /// @warning Dimension of the input is not checked
    void forward_(const blitz::Array<double,1>& input, double& output) const;
    
    /// Accumulates the GMM statistics over a set of samples.
    /// @see bool accStatistics(const blitz::Array<double,1> &x, GMMStats stats)
    void accStatistics(const Torch::io::Arrayset &arrayset, GMMStats &stats) const;

    /// Accumulate the GMM statistics for this sample.
    ///
    /// @param[in]  x     The current sample
    /// @param[out] stats The accumulated statistics
    void accStatistics(const blitz::Array<double,1> &x, GMMStats &stats) const;
    
    /// Get the weights
    /// @param[out] weights The weights ("mixing coefficients") of the Gaussian components
    void getWeights(blitz::Array<double,1> &weights) const;

    /// Get a pointer to a particular Gaussian component
    /// @param[i] i The index of the Gaussian component
    /// @return A pointer to the i'th Gaussian component
    ///         if it exists, otherwise NULL
    Gaussian* getGaussian(int i) const;

    /// Return the number of Gaussian components
    int getNGaussians() const;

    /// Save to a Configuration
    void save(Torch::io::HDF5File& config) const;
    
    /// Load from a Configuration
    void load(Torch::io::HDF5File& config);

    /// Load/Reload mean/variance supervector in cache
    void reloadCacheSupervectors() const;
    
    friend std::ostream& operator<<(std::ostream& os, const GMMMachine& machine);
    
  protected:

    /// Copy another GMMMachine
    void copy(const GMMMachine&);
    
    /// The number of Gaussian components
    int64_t m_n_gaussians;
    
    /// The feature dimensionality
    int64_t m_n_inputs;

    /// The Gaussian components
    Gaussian *m_gaussians;

    /// The weights (also known as "mixing coefficients")
    blitz::Array<double,1> m_weights;

  private:

    /// Update the mean and variance supervectors 
    /// in cache (into a 1D blitz array)
    void updateCacheSupervectors() const;

    /// Some cache arrays to avoid re-allocation when computing log-likelihoods
    mutable blitz::Array<double,1> m_cache_log_weighted_gaussian_likelihoods;
    mutable blitz::Array<double,1> m_cache_P;
    mutable blitz::Array<double,2> m_cache_Px;
    mutable blitz::Array<double,2> m_cache_Pxx;

    mutable blitz::Array<double,1> m_cache_mean_supervector;
    mutable blitz::Array<double,1> m_cache_variance_supervector;
    mutable bool m_cache_supervector;
    
};

}}

#endif
