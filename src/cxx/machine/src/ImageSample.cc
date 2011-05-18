#include <machine/ImageSample.h>

Torch::machine::ImageSample::~ImageSample () {
  
}

Torch::machine::ImageSample::ImageSample(const ImageSample& copy) : array(copy.array) {
  
}

Torch::machine::ImageSample::ImageSample(const blitz::Array<double, 2>& array): array(array) {
  
}

const blitz::Array<double, 2>& Torch::machine::ImageSample::getImage() const {
  return array;
}
