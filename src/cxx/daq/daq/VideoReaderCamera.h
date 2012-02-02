#ifndef VIDEOREADERCAMERA_H
#define VIDEOREADERCAMERA_H

#include <daq/Camera.h>
#include <io/VideoReader.h>

namespace bob { namespace daq {

/**
 * Read a video file from a bob::io::VideoReader
 */
class VideoReaderCamera : public Camera {
public:

  VideoReaderCamera(boost::shared_ptr<bob::io::VideoReader> videoReader);
  ~VideoReaderCamera();
  
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
  pthread_t thread;
  bool mustStop;
  
  boost::shared_ptr<bob::io::VideoReader> videoReader;

  blitz::Array<uint8_t, 3> bobFrame;
  blitz::Array<uint8_t, 3> frame;
};

}}

#endif // VIDEOREADERCAMERA_H
