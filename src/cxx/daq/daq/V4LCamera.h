#ifndef V4LCAMERA_H
#define V4LCAMERA_H

#include <daq/Camera.h>

namespace bob { namespace daq {
  
struct V4LStruct;

/**
 * Capture frames using Video for Linux 2
 */
class V4LCamera:public Camera {
public:
  /**
   * @param device path to the video device (e.g. "/dev/video0")
   */
  V4LCamera(const char* device);
  virtual ~V4LCamera();

  int open();
  void close();
  int start();
  void stop();
  void wait();

  int getSupportedPixelFormats(std::vector<PixelFormat>& pixelFormats);
  int getSupportedFrameSizes(PixelFormat pixelFormat, std::vector<FrameSize>& frameSizes);
  int getSupportedFrameIntervals(PixelFormat pixelFormat, FrameSize& frameSize, std::vector<FrameInterval>& frameIntervals);

  PixelFormat getPixelFormat() const;
  void setPixelFormat(PixelFormat pixelFormat);
  
  FrameSize getFrameSize() const;
  void setFrameSize(FrameSize& frameSize);

  FrameInterval getFrameInterval() const;
  void setFrameInterval(FrameInterval& frameInterval);

  void printSummary();


  void captureLoop();
  
private:
  V4LStruct* v4lstruct;
  bool mustStop;
};

}}
#endif // V4LCAMERA_H
