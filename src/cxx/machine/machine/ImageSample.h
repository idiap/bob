#ifndef TORCH5SPRO_MACHINE_IMAGESAMPLE_H
#define TORCH5SPRO_MACHINE_IMAGESAMPLE_H
#include <blitz/array.h>

namespace Torch {
namespace machine {

/**
 * This class represents one Image. It encapsulate a blitz::Array<double, 1>
 */
class ImageSample {
public:
  
  virtual ~ImageSample();

  /// Constructor
  ImageSample(const blitz::Array<double, 2>& array);
  
  /// Copy constructor
  ImageSample(const ImageSample& copy);

  /// Get the Image
  const blitz::Array<double, 2>& getImage() const; 
  
private:
  blitz::Array<double, 2> array;
};
}
}

#endif // TORCH5SPRO_MACHINE_IMAGESAMPLE_H
