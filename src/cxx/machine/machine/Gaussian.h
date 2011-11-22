/**
 * @file cxx/machine/machine/Gaussian.h
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
/// @file Gaussian.h
/// @author <a href="mailto:Roy.Wallace@idiap.ch">Roy Wallace</a> 
/// @brief This class implements a multivariate diagonal Gaussian distribution. 

#ifndef _GAUSSIAN_H
#define _GAUSSIAN_H

#include <blitz/array.h>
#include <cfloat>
#include <io/HDF5File.h>

namespace Torch {
namespace machine {

namespace Log
{
  #define MINUS_LOG_THRESHOLD -39.14
  
  const double Log2Pi = 1.83787706640934548355;
  const double LogZero = -DBL_MAX;
  const double LogOne = 0;
  
  double LogAdd(double log_a, double log_b);
  double LogSub(double log_a, double log_b);
}

/// @brief This class implements a multivariate diagonal Gaussian distribution. 
class Gaussian {
  public:

    /// Default constructor
    Gaussian();

    /// Constructor
    /// @param[in] n_inputs The feature dimensionality
    Gaussian(int n_inputs);

    /// Constructor
    Gaussian(Torch::io::HDF5File& config);

    /// Destructor
    virtual ~Gaussian();

    /// Copy constructor
    Gaussian(const Gaussian& other);

    /// Assigment
    Gaussian& operator= (const Gaussian &other);

    /// Equal to
    bool operator ==(const Gaussian& b) const;
    
    /// Set the input dimensionality, reset the mean to zero
    /// and the variance to one.
    /// @see resize()
    /// @param n_inputs The feature dimensionality
    void setNInputs(int n_inputs);

    /// Get the input dimensionality
    int getNInputs();
    
    /// Set the input dimensionality, reset the mean to zero
    /// and the variance to one.
    /// @see setNInputs()
    /// @param n_inputs The feature dimensionality
    void resize(int n_inputs);
    
    /// Get the mean
    void getMean(blitz::Array<double,1> &mean) const;
    
    /// Set the mean
    void setMean(const blitz::Array<double,1> &mean);

    /// Get the variance (the diagonal of the covariance matrix)
    void getVariance(blitz::Array<double,1> &variance) const;
    
    /// Set the variance
    void setVariance(const blitz::Array<double,1> &variance);

    /// Get the variance flooring thresholds
    void getVarianceThresholds(blitz::Array<double,1> &variance_thresholds) const;
    
    /// Set the variance flooring thresholds
    void setVarianceThresholds(const blitz::Array<double,1> &variance_thresholds);

    /// Set the variance flooring thresholds
    void setVarianceThresholds(double factor);

    /// Output the log likelihood of the sample, x 
    /// @param x The data sample (feature vector)
    double logLikelihood(const blitz::Array<double, 1> &x) const;

    /// Save to a Configuration
    void save(Torch::io::HDF5File& config) const;
    
    /// Load from a Configuration
    void load(Torch::io::HDF5File& config);

    friend std::ostream& operator<<(std::ostream& os, const Gaussian& g);
    
  protected:

    /// Copy another Gaussian
    void copy(const Gaussian& other);
    
    /// Compute and store the value of g_norm, 
    /// to later speed up evaluation of logLikelihood()
    /// Note: g_norm is defined as follows:
    /// log(Gaussian pdf) = log(1/((2pi)^(k/2)(det)^(1/2)) * exp(...))
    ///                   = -1/2 * g_norm * (...)
    void preComputeConstants();

    /// The mean
    blitz::Array<double,1> m_mean;

    /// The diagonal of the covariance matrix
    blitz::Array<double,1> m_variance;

    /// The variance flooring thresholds, i.e. the minimum allowed
    /// value of variance in each dimension.
    /// The variance will be set to this value if an attempt is made
    /// to set it to a smaller value.
    blitz::Array<double,1> m_variance_thresholds;

    /// A constant that depends only on the feature dimensionality
    /// (m_n_inputs) and the variance
    /// @see bool preComputeConstants()
    double g_norm;

    /// The number of inputs
    int64_t m_n_inputs;
};


}
}
#endif
