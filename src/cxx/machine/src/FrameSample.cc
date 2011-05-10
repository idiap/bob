#include <machine/FrameSample.h>

Torch::machine::FrameSample::~FrameSample () {
  
}

Torch::machine::FrameSample::FrameSample(const FrameSample& copy) : array(copy.array) {
  
}

Torch::machine::FrameSample::FrameSample(const blitz::Array<float, 1>& array): array(array) {
  
}

const blitz::Array<float, 1>& Torch::machine::FrameSample::getFrame() const {
  return array;
}
