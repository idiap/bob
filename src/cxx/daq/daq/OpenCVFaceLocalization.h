#ifndef OPENCVFACELOCALIZATION_H
#define OPENCVFACELOCALIZATION_H

#include <daq/FaceLocalization.h>

#include <cv.h>

namespace bob { namespace daq {

/**
 * Provide face localization using OpenCV
 */
class OpenCVFaceLocalization:public FaceLocalization
{
public:
  OpenCVFaceLocalization(const char* model_path);
  virtual ~OpenCVFaceLocalization();
    
  virtual void imageReceived (blitz::Array<unsigned char, 2>& image, CaptureStatus& status);
  virtual bool start();
  virtual void stop();
  
  void localize();
  
private:
  pthread_t thread;
  
  cv::Mat img;
  pthread_mutex_t img_mutex;

  CvHaarClassifierCascade* cascade;

  bool mustStop;
};

}}

#endif // OPENCVFACELOCALIZATION_H
