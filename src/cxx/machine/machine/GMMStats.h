/**
 * @file cxx/machine/machine/GMMStats.h
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
/// @file GMMStats.h
/// @author <a href="mailto:Roy.Wallace@idiap.ch">Roy Wallace</a> 
/// @author <a href="mailto:Laurent.El-Shafey@idiap.ch">Laurent El Shafey</a> 

#ifndef TORCH5SPRO_MACHINE_GMMSTATS_H
#define TORCH5SPRO_MACHINE_GMMSTATS_H

#include <blitz/array.h>
#include "io/HDF5File.h"

namespace Torch {
namespace machine {

/// @brief A container for GMM statistics.
/// @see GMMMachine
///
/// With respect to Reynolds, "Speaker Verification Using Adapted
/// Gaussian Mixture Models", DSP, 2000:
/// Eq (8) is n(i)
/// Eq (9) is sumPx(i) / n(i)
/// Eq (10) is sumPxx(i) / n(i)
class GMMStats {
  public:
    
    /// Default constructor.
    GMMStats();

    /// Constructor.
    /// @param n_gaussians Number of Gaussians in the mixture model.
    /// @param n_inputs    Feature dimensionality.
    GMMStats(int n_gaussians, int n_inputs);

    /// Copy constructor
    GMMStats(const GMMStats& other);

    /// Constructor
    GMMStats(Torch::io::HDF5File& config);
    
    /// Assigment
    GMMStats& operator=(const GMMStats& other);

    /// Equal to
    bool operator==(const GMMStats& b) const;

    /// Destructor
    ~GMMStats();

    /// Allocates space for the statistics and resets to zero.
    /// @param n_gaussians Number of Gaussians in the mixture model.
    /// @param n_inputs    Feature dimensionality.
    void resize(int n_gaussians, int n_inputs);

    /// Resets statistics to zero.
    void init();
 
    /// The accumulated log likelihood of all samples
    double log_likelihood;

    /// The accumulated number of samples
    int64_t T;

    /// For each Gaussian, the accumulated sum of responsibilities, i.e. the sum of P(gaussian_i|x)
    blitz::Array<double,1> n;

    /// For each Gaussian, the accumulated sum of responsibility times the sample 
    blitz::Array<double,2> sumPx;

    /// For each Gaussian, the accumulated sum of responsibility times the sample squared
    blitz::Array<double,2> sumPxx;

    /// Save to a Configuration
    void save(Torch::io::HDF5File& config) const;
    
    /// Load from a Configuration
    void load(Torch::io::HDF5File& config);
    
    friend std::ostream& operator<<(std::ostream& os, const GMMStats& g);

  protected:
    /// Copy another GMMStats
    void copy(const GMMStats&);
};

}
}

#endif // _GMMSTATS_H
