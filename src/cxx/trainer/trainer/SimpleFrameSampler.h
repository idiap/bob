#ifndef SIMPLEFRAMESAMPLER_H
#define SIMPLEFRAMESAMPLER_H
#include <io/Arrayset.h>
#include <trainer/Sampler.h>
#include <machine/FrameSample.h>


namespace Torch {
namespace trainer {

/**
 * This class provides a list of FrameSample from an arrayset
 */
class SimpleFrameSampler : public Sampler<Torch::machine::FrameSample> {
public:
  
  virtual ~SimpleFrameSampler() {}

  /// Constructor
  SimpleFrameSampler(const Torch::io::Arrayset& arrayset);

  /// Copy constructor
  SimpleFrameSampler(const SimpleFrameSampler& other);

  const Torch::machine::FrameSample getSample(int index) const; 
  
  int getNSamples() const; 
  
private:
  Torch::io::Arrayset arrayset;
};

}
}

#endif // SIMPLEFRAMESAMPLER_H
