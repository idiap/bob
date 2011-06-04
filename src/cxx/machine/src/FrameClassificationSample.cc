#include <machine/FrameClassificationSample.h>

Torch::machine::FrameClassificationSample::~FrameClassificationSample() 
{
}

Torch::machine::FrameClassificationSample::FrameClassificationSample(const FrameClassificationSample& copy):
  m_array(copy.m_array), m_target(copy.m_target) 
{  
}

Torch::machine::FrameClassificationSample::FrameClassificationSample(const blitz::Array<double, 1>& array, const uint32_t target): 
  m_array(array), m_target(target) 
{  
}

const blitz::Array<double, 1>& 
Torch::machine::FrameClassificationSample::getFrame() const {
  return m_array;
}

int Torch::machine::FrameClassificationSample::getFrameSize() const {
  return m_array.extent(0);
}

uint32_t Torch::machine::FrameClassificationSample::getTarget() const {
  return m_target;
}
