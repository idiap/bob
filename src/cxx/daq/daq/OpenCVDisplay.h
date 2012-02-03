#ifndef OPENCVDISPLAY_H
#define OPENCVDISPLAY_H

#include <daq/Display.h>
#include <cv.h>

namespace bob { namespace daq {

/**
 * Display a GUI using OpenCV
 */
class OpenCVDisplay :public Display {
public:
  OpenCVDisplay();
  virtual ~OpenCVDisplay();

  void imageReceived(blitz::Array<unsigned char, 2>& image, CaptureStatus& status);
  void onDetection(FaceLocalizationCallback::BoundingBox& bb);

  void start();
  void stop();
  
private:
  cv::Mat image;
  pthread_mutex_t image_mutex;
  CaptureStatus captureStatus;
  
  FaceLocalizationCallback::BoundingBox boundingBox;
  pthread_mutex_t boundingBox_mutex;

  bool mustStop;
  
  double fps;
  int fps_nbFrame;
  double fps_startTime;
  
};

}}

#endif // OPENCVDISPLAY_H
