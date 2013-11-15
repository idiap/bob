/**
 * @file bob/machine/GMMStats.h
 * @date Tue May 10 11:35:58 2011 +0200
 * @author Francois Moulin <Francois.Moulin@idiap.ch>
 * @author Laurent El Shafey <Laurent.El-Shafey@idiap.ch>
 *
 * Copyright (C) 2011-2013 Idiap Research Institute, Martigny, Switzerland
 */

#ifndef BOB_MACHINE_GMMSTATS_H
#define BOB_MACHINE_GMMSTATS_H

#include <blitz/array.h>
#include <bob/io/HDF5File.h>

namespace bob { namespace machine {
/**
 * @ingroup MACHINE
 * @{
 */

/**
 * @brief A container for GMM statistics.
 * @see GMMMachine
 *
 * With respect to Reynolds, "Speaker Verification Using Adapted
 * Gaussian Mixture Models", DSP, 2000:
 * Eq (8) is n(i)
 * Eq (9) is sumPx(i) / n(i)
 * Eq (10) is sumPxx(i) / n(i)
 */
class GMMStats {
  public:
    
    /**
     * Default constructor.
     */
    GMMStats();

    /**
     * Constructor.
     * @param n_gaussians Number of Gaussians in the mixture model.
     * @param n_inputs    Feature dimensionality.
     */
    GMMStats(const size_t n_gaussians, const size_t n_inputs);

    /**
     * Copy constructor
     */
    GMMStats(const GMMStats& other);

    /**
     * Constructor (from a Configuration)
     */
    GMMStats(bob::io::HDF5File& config);
    
    /**
     * Assigment
     */
    GMMStats& operator=(const GMMStats& other);

    /**
     * Equal to
     */
    bool operator==(const GMMStats& b) const;

    /**
     * Not Equal to
     */
    bool operator!=(const GMMStats& b) const;

    /**
     * @brief Similar to
     */
    bool is_similar_to(const GMMStats& b, const double r_epsilon=1e-5,
      const double a_epsilon=1e-8) const;

    /**
     * Updates a GMMStats with another GMMStats
     */
    void operator+=(const GMMStats& b);

    /**
     * Destructor
     */
    ~GMMStats();

    /**
     * Allocates space for the statistics and resets to zero.
     * @param n_gaussians Number of Gaussians in the mixture model.
     * @param n_inputs    Feature dimensionality.
     */
    void resize(const size_t n_gaussians, const size_t n_inputs);

    /**
     * Resets statistics to zero.
     */
    void init();
 
    /**
     * The accumulated log likelihood of all samples
     */
    double log_likelihood;

    /**
     * The accumulated number of samples
     */
    size_t T;

    /**
     * For each Gaussian, the accumulated sum of responsibilities, i.e. the sum of P(gaussian_i|x)
     */
    blitz::Array<double,1> n;

    /**
     * For each Gaussian, the accumulated sum of responsibility times the sample 
     */
    blitz::Array<double,2> sumPx;

    /**
     * For each Gaussian, the accumulated sum of responsibility times the sample squared
     */
    blitz::Array<double,2> sumPxx;

    /**
     * Save to a Configuration
     */
    void save(bob::io::HDF5File& config) const;
    
    /**
     * Load from a Configuration
     */
    void load(bob::io::HDF5File& config);
    
    friend std::ostream& operator<<(std::ostream& os, const GMMStats& g);

  private:
    /**
     * Copy another GMMStats
     */
    void copy(const GMMStats&);
};

/**
 * @}
 */
}}

#endif 
