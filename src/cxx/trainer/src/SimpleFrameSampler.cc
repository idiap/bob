#include "trainer/SimpleFrameSampler.h"

Torch::trainer::SimpleFrameSampler::SimpleFrameSampler(const Torch::database::Arrayset& arrayset) : 
  arrayset(arrayset) {
  
}

Torch::trainer::SimpleFrameSampler::SimpleFrameSampler(const SimpleFrameSampler& other) : 
  arrayset(other.arrayset) {
  
}

const Torch::machine::FrameSample Torch::trainer::SimpleFrameSampler::getSample(int index) const {
  return Torch::machine::FrameSample(arrayset[index+1].cast<double, 1>());
}

int Torch::trainer::SimpleFrameSampler::getNSamples() const {
  return arrayset.getNSamples();
}