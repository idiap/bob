#ifndef SAMPLER_H
#define SAMPLER_H

#include <boost/shared_ptr.hpp>

namespace Torch {
namespace trainer {

/**
 * This class provides a list of Sample.
 */
template<class T_sample>
class Sampler {
public:
  
  virtual ~Sampler() {}

  /**
   * Get a sample
   *
   * @param index index of the sample (0 <= index < getNSamples())
   * @return a Sample
   */
  virtual const T_sample getSample(int index) const = 0;

  /**
   * Get the number of Samples
   */
  virtual int getNSamples() const = 0;
};

}
}

#endif // SAMPLER_H
