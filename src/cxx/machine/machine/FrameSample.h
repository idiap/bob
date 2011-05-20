#ifndef FRAMESAMPLE_H
#define FRAMESAMPLE_H
#include <blitz/array.h>

namespace Torch {
namespace machine {

/**
 * This class represents one Frame. It encapsulates a blitz::Array<double, 1>
 */
class FrameSample {
public:
  
  virtual ~FrameSample();

  /// Constructor
  FrameSample(const blitz::Array<double, 1>& array);
  
  /// Copy constructor
  FrameSample(const FrameSample& copy);

  /// Get the Frame
  const blitz::Array<double, 1>& getFrame() const;

  /// Get the frame size
  int getFrameSize() const;
  
private:
  blitz::Array<double, 1> array;
  const int frameSize;
};
}
}

#endif // FRAMESAMPLE_H