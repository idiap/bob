#include <machine/FrameSample.h>

Torch::machine::FrameSample::~FrameSample () {
  
}

Torch::machine::FrameSample::FrameSample(const FrameSample& copy) : array(copy.array), frameSize(copy.frameSize) {
  
}

Torch::machine::FrameSample::FrameSample(const blitz::Array<double, 1>& array): array(array), frameSize(array.extent(0)) {
  
}

const blitz::Array<double, 1>& Torch::machine::FrameSample::getFrame() const {
  return array;
}

int Torch::machine::FrameSample::getFrameSize() const {
  return frameSize;
}