#ifndef VISIONERFACELOCALIZATION_H
#define VISIONERFACELOCALIZATION_H

#include <daq/FaceLocalization.h>

namespace bob { namespace daq {

struct Visioner_ptr;

/**
 * Provide face localization using Visioner
 */
class VisionerFaceLocalization : public FaceLocalization {

public:
  /**
   * @param model_path path to a model file (e.g. Face.MCT9.gz)
   */
  VisionerFaceLocalization(const char* model_path);
  virtual ~VisionerFaceLocalization();
  
  void imageReceived(blitz::Array<unsigned char, 2>& image, CaptureStatus& status);

  void stop();
  bool start();

  
  void localize();
private:
  pthread_t thread;
  
  Visioner_ptr* visioner_ptr;
  
  blitz::Array<unsigned char, 2> img;
  pthread_mutex_t img_mutex;
  CaptureStatus status;

  int lastid;

  bool mustStop;
};

}}
#endif // VISIONERFACELOCALIZATION_H
