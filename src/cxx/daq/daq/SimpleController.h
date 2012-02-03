#ifndef SIMPLECONTROLER_H
#define SIMPLECONTROLER_H

#include "daq/Controller.h"

namespace bob { namespace daq {

/**
 * Default Controller implementation
 */
class SimpleController : public Controller {

public:
  SimpleController();
  virtual ~SimpleController();
  
  virtual void keyPressed(int key);
  virtual void imageReceived(unsigned char* image, Camera::PixelFormat pixelformat, int width, int height, int stride, int size, int frameNb, double timestamp);

  void stop();
  
private:
  double firstFrameTimestamp;
  int firstRecordingFrameNumber;
  double firstRecordingFrameTimestamp;
  bool recording;

  unsigned char* buffer;
  int bufferSize;
};

}}
#endif // SIMPLECONTROLER_H
