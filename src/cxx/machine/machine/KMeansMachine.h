#ifndef _KMEANSMACHINE_H
#define _KMEANSMACHINE_H

#include <blitz/array.h>
#include <cfloat>

#include "io/Arrayset.h"
#include "machine/Machine.h"

namespace Torch {
namespace machine {

/// @brief This class implements a k-means classifier.
/// @details See Section 9.1 of Bishop, "Pattern recognition and machine learning", 2006
class KMeansMachine : public Machine<blitz::Array<double,1>, double> {
public:
  
  /// Constructor
  /// @param[in] n_means  The number of means
  /// @param[in] n_inputs The feature dimensionality
  KMeansMachine(int n_means, int n_inputs); 
  
  /// Destructor
  virtual ~KMeansMachine();
  
  /// Set the means
  void setMeans(const blitz::Array<double,2> &means);
  
  /// Set the i'th mean
  void setMean(int i, const blitz::Array < double, 1 > & mean);
  
  /// Get a mean
  /// @param[i]   i    The index of the mean
  /// @param[out] mean The mean, a 1D array, with a length equal to the number of feature dimensions.
  void getMean(int i, blitz::Array <double, 1> & mean) const;
  
  /// Get the means
  /// @param[out] means A 2D array, with as many rows as means, and as many columns as feature dimensions.
  void getMeans(blitz::Array <double, 2> & means) const;
  
  /// Return the Euclidean distance of the sample, x, 
  /// to the i'th mean
  /// @param x The data sample (feature vector)
  /// @param i The index of the mean
  double getDistanceFromMean(const blitz::Array<double,1> &x, int i) const;
  
  /// Calculate the index of the mean that is closest
  /// (in terms of Euclidean distance) to the data sample, x
  /// @param x The data sample (feature vector)
  /// @param closest_mean (output) The index of the mean closest to the sample
  /// @param distance (output) The distance of the sample from the closest mean
  void getClosestMean(const blitz::Array<double,1> &x, int &closest_mean, double &min_distance) const;
  
  /// Output the minimum distance between the input and one of the means
  double getMinDistance(const blitz::Array <double, 1> & input) const;

  /// For each mean, find the subset of the samples
  /// that is closest to that mean, and calculate
  /// 1) the variance of that subset (the cluster variance)
  /// 2) the proportion of the samples represented by that subset (the cluster weight)
  /// @param[in]  sampler   The sampler
  /// @param[out] variances The cluster variances (one row per cluster), with as many columns as feature dimensions.
  /// @param[out] weights   A vector of weights, one per cluster
  void getVariancesAndWeightsForEachCluster(const Torch::io::Arrayset &ar, blitz::Array<double,2> &variances, blitz::Array<double,1> &weights) const;
  
  /// Output the minimum distance between the input and one of the means
  /// (overrides Machine::forward)
  void forward(const blitz::Array<double,1>& input, double& output) const;
  
  /// Output the minimum distance between the input and one of the means
  /// (overrides Machine::forward_)
  /// @warning Inputs are NOT checked
  void forward_(const blitz::Array<double,1>& input, double& output) const;
  
  /// Return the number of means
  int getNMeans() const;
  
  /// Return the number of inputs
  int getNInputs() const;
  
protected:
  
  /// The number of means
  int m_n_means;
  
  /// The number of inputs
  int m_n_inputs;
  
  /// The means (each row is a mean)
  blitz::Array<double,2> m_means;

private:
  /// cache to avoid re-allocation
  mutable blitz::Array<double,2> m_cache_means;
};

}
}
#endif // KMEANSMACHINE_H
