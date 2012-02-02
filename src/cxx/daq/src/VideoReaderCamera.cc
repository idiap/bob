#include "daq/VideoReaderCamera.h"
#include <io/Array.h>

//static pthread_mutex_t pthread_mutex_initializer = PTHREAD_MUTEX_INITIALIZER;

namespace bob { namespace daq {
  
VideoReaderCamera::VideoReaderCamera(boost::shared_ptr<bob::io::VideoReader> videoReader) {
  assert(videoReader != NULL);
  this->videoReader = videoReader;
  mustStop = false;
  thread = 0;
}

VideoReaderCamera::~VideoReaderCamera() {
  
}

int VideoReaderCamera::open() {
  return 0;
}

void VideoReaderCamera::close() {
  stop();
}

static void* captureLoop_(void* param) {
  VideoReaderCamera* me = (VideoReaderCamera*)param;
  me->captureLoop();

  return NULL;
}

int VideoReaderCamera::start() {
  mustStop = false;
  
  // Start the thread
  int error = pthread_create(&thread, NULL, captureLoop_, (void*) this);

  if (error != 0) {
    return -1;
  }
  return 0;
}

void VideoReaderCamera::captureLoop() {
  bob::io::VideoReader::const_iterator itframe = videoReader->begin();

  bobFrame.resize(3, videoReader->height(), videoReader->width());
  frame.resize(bobFrame.extent(1), bobFrame.extent(2), bobFrame.extent(0));
  
  while(!mustStop) {
    if (itframe == videoReader->end()) {
      break;
    }
    
    size_t frameNb = itframe.cur();
    itframe.read(bobFrame);
    frame = bobFrame.transpose(1, 2, 0);

    pthread_mutex_lock(&callbacks_mutex);
    for(std::vector<CameraCallback*>::iterator it = callbacks.begin(); it != callbacks.end(); it++) {
      (*it)->imageReceived(frame.data(), RGB24,
                           frame.extent(1), frame.extent(0),
                           frame.extent(1)*sizeof(unsigned int),
                           frame.extent(0)*frame.extent(1)*sizeof(unsigned int),
                           frameNb, frameNb*(1/videoReader->frameRate()));
    }
    pthread_mutex_unlock(&callbacks_mutex);
  }
}

void VideoReaderCamera::stop() {
  mustStop = true;
  wait();
}

void VideoReaderCamera::wait() {
  if (thread != 0) {
    pthread_join(thread, NULL);
  }
}

void VideoReaderCamera::printSummary() {
  printf("%s\n", videoReader->info().c_str());
}

Camera::PixelFormat VideoReaderCamera::getPixelFormat() const {
  return RGB24;
}

void VideoReaderCamera::setPixelFormat(Camera::PixelFormat pixelFormat) {
  return;
}

Camera::FrameSize VideoReaderCamera::getFrameSize() const {
  return FrameSize(videoReader->width(), videoReader->height());
}

void VideoReaderCamera::setFrameSize(Camera::FrameSize& frameSize) {
  return;
}

Camera::FrameInterval VideoReaderCamera::getFrameInterval() const {
  return FrameInterval(1, (int)videoReader->frameRate());
}

void VideoReaderCamera::setFrameInterval(Camera::FrameInterval& frameInterval) {
  return;
}

int VideoReaderCamera::getSupportedPixelFormats(std::vector<PixelFormat>& pixelFormats) {
  pixelFormats.clear();;
  pixelFormats.push_back(getPixelFormat());

  return 0;
}

int VideoReaderCamera::getSupportedFrameSizes(PixelFormat pixelFormat, std::vector<FrameSize>& frameSizes) {
  frameSizes.clear();
  if (pixelFormat == getPixelFormat()) {
    frameSizes.push_back(getFrameSize());
  }

  return 0;
}

int VideoReaderCamera::getSupportedFrameIntervals(PixelFormat pixelFormat, FrameSize& frameSize,
                                         std::vector<FrameInterval>& frameIntervals) {
  frameIntervals.clear();
  if (pixelFormat == getPixelFormat() && frameSize == getFrameSize()) {
    frameIntervals.push_back(getFrameInterval());
  }
  
  return 0;
}

}}
