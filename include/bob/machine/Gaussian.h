/**
 * @file bob/machine/Gaussian.h
 * @date Tue May 10 11:35:58 2011 +0200
 * @author Francois Moulin <Francois.Moulin@idiap.ch>
 * @author Laurent El Shafey <Laurent.El-Shafey@idiap.ch>
 *
 * Copyright (C) 2011-2013 Idiap Research Institute, Martigny, Switzerland
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

#ifndef BOB_MACHINE_GAUSSIAN_H
#define BOB_MACHINE_GAUSSIAN_H

#include "bob/machine/Machine.h"
#include "bob/io/HDF5File.h"
#include <blitz/array.h>
#include <limits>

namespace bob { namespace machine {

/**
 * @brief This class implements a multivariate diagonal Gaussian distribution.
 */
class Gaussian: public Machine<blitz::Array<double,1>, double>
{
  public:
    /**
     * Default constructor
     */
    Gaussian();

    /**
     * Constructor
     * @param[in] n_inputs The feature dimensionality
     */
    Gaussian(const size_t n_inputs);

    /**
     * Destructor
     */
    virtual ~Gaussian();

    /**
     * Copy constructor
     */
    Gaussian(const Gaussian& other);

    /**
     * Constructs from a configuration file
     */
    Gaussian(bob::io::HDF5File& config);

    /**
     * Assignment
     */
    Gaussian& operator=(const Gaussian &other);

    /**
     * Equal to
     */
    bool operator==(const Gaussian& b) const;
    /**
     * Not equal to
     */
    bool operator!=(const Gaussian& b) const;

    /**
     * Similar to
     */
    bool is_similar_to(const Gaussian& b, const double epsilon=1e-8) const;

    /**
     * Set the input dimensionality, reset the mean to zero
     * and the variance to one.
     * @see resize()
     * @param n_inputs The feature dimensionality
     * @warning The mean and variance are not initialized
     */
    void setNInputs(const size_t n_inputs);

    /**
     * Get the input dimensionality
     */
    size_t getNInputs() const
    { return m_n_inputs; }

    /**
     * Set the input dimensionality, reset the mean to zero
     * and the variance to one.
     * @see setNInputs()
     * @param n_inputs The feature dimensionality
     */
    void resize(const size_t n_inputs);

    /**
     * Get the mean
     */
    inline const blitz::Array<double,1>& getMean() const
    { return m_mean; }

    /**
     * Get the mean in order to be updated
     * @warning Only trainers should use this function for efficiency reason
     */
    inline blitz::Array<double,1>& updateMean()
    { return m_mean; }

    /**
     * Set the mean
     */
    void setMean(const blitz::Array<double,1> &mean);

    /**
     * Get the variance (the diagonal of the covariance matrix)
     */
    inline const blitz::Array<double,1>& getVariance() const
    { return m_variance; }

    /**
     * Get the variance in order to be updated
     * @warning Only trainers should use this function for efficiency reason
     */
    inline blitz::Array<double,1>& updateVariance()
    { return m_variance; }

    /**
     * Set the variance
     */
    void setVariance(const blitz::Array<double,1> &variance);

    /**
     * Get the variance flooring thresholds
     */
    const blitz::Array<double,1>& getVarianceThresholds() const
    { return m_variance_thresholds; }

    /**
     * Get the variance thresholds in order to be updated
     * @warning Only trainers should use this function for efficiency reason
     */
    inline blitz::Array<double,1>& updateVarianceThreshods()
    { return m_variance_thresholds; }

    /**
     * Set the variance flooring thresholds
     */
    void setVarianceThresholds(const blitz::Array<double,1> &variance_thresholds);

    /**
     * Set the variance flooring thresholds
     */
    void setVarianceThresholds(const double value);

    /**
     * Apply the variance flooring thresholds
     * This method is called when using setVarianceThresholds()
     * @warning It is only useful when using updateVarianceThreshods(),
     * and should mostly be done by trainers
     */
    void applyVarianceThresholds();

    /**
     * Output the log likelihood of the sample, x
     * @param x The data sample (feature vector)
     */
    double logLikelihood(const blitz::Array<double,1>& x) const;

    /**
     * Output the log likelihood of the sample, x
     * @param x The data sample (feature vector)
     * @warning The input is NOT checked
     */
    double logLikelihood_(const blitz::Array<double,1>& x) const;

    /**
     * Computes the log likelihood of the sample, x
     * @param x The data sample (feature vector)
     * @param output The computed log likelihood
     */
    void forward(const blitz::Array<double,1>& x, double& output) const
    { output = logLikelihood(x); }

    /**
     * Computes the log likelihood of the sample, x
     * @param x The data sample (feature vector)
     * @param output The computed log likelihood
     * @warning The input is NOT checked
     */
    void forward_(const blitz::Array<double,1>& x, double& output) const
    { output = logLikelihood_(x); }

    /**
     * Saves to a Configuration
     */
    void save(bob::io::HDF5File& config) const;

    /**
     * Loads from a Configuration
     */
    void load(bob::io::HDF5File& config);

    /**
     * Prints a Gaussian in the output stream
     */
    friend std::ostream& operator<<(std::ostream& os, const Gaussian& g);


  private:
    /**
     * Copies another Gaussian
     */
    void copy(const Gaussian& other);

    /**
     * Computes n_inputs * log(2*pi)
     */
    void preComputeNLog2Pi();

    /**
     * Computes and stores the value of g_norm,
     * to later speed up evaluation of logLikelihood()
     * Note: g_norm is defined as follows:
     * log(Gaussian pdf) = log(1/((2pi)^(k/2)(det)^(1/2)) * exp(...))
     *                   = -1/2 * g_norm * (...)
     */
    void preComputeConstants();

    /**
     * The mean vector of the Gaussian
     */
    blitz::Array<double,1> m_mean;

    /**
     * The diagonal of the covariance matrix (assumed to be diagonal)
     */
    blitz::Array<double,1> m_variance;

    /**
     * The variance flooring thresholds, i.e. the minimum allowed
     * value of variance in each dimension.
     * The variance will be set to this value if an attempt is made
     * to set it to a smaller value.
     */
    blitz::Array<double,1> m_variance_thresholds;

    /**
     * A constant that depends only on the feature dimensionality
     * m_n_log2pi = n_inputs * log(2*pi) (used to compute m_gnorm)
     */
    double m_n_log2pi;

    /**
     * A constant that depends only on the feature dimensionality
     * (m_n_inputs) and the variance
     * @see bool preComputeConstants()
     */
    double m_g_norm;

    /**
     * The number of inputs (feature dimensionality)
     */
    size_t m_n_inputs;
};

}}
#endif
