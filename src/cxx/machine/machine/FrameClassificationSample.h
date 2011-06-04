/**
 * @file src/cxx/math/math/FrameClassificationSample.h
 * @author <a href="mailto:Laurent.El-Shafey@idiap.ch">Laurent El Shafey</a> 
 *
 * @brief This file defines a sample with a target for dealing with 
 *  classification problems.
 * 
 */

#ifndef TORCH5SPRO_MACHINE_FRAME_CLASSIFICATION_SAMPLE_H
#define TORCH5SPRO_MACHINE_FRAME_CLASSIFICATION_SAMPLE_H

#include <blitz/array.h>
#include <stdint.h> // for uint32_t declaration/typedef

namespace Torch {
namespace machine {

/**
 * This class represents one Frame. It encapsulates a blitz::Array<double, 1> 
 * as well as a uint32_t target id
 */
class FrameClassificationSample {
public:
  
  virtual ~FrameClassificationSample();

  /// Constructor
  FrameClassificationSample(const blitz::Array<double, 1>& array, 
    const uint32_t target);
  
  /// Copy constructor
  FrameClassificationSample(const FrameClassificationSample& copy);

  /// Get the Frame
  const blitz::Array<double, 1>& getFrame() const;

  /// Get the Target
  uint32_t getTarget() const;

  /// Get the frame size
  int getFrameSize() const;
  
private:
  blitz::Array<double, 1> m_array;
  const uint32_t m_target;
};
}
}

#endif /* TORCH5SPRO_MACHINE_FRAME_CLASSIFICATION_SAMPLE_H */
