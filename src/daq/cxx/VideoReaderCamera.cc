/**
 * @file daq/cxx/VideoReaderCamera.cc
 * @date Thu Feb 2 11:22:57 2012 +0100
 * @author Francois Moulin <Francois.Moulin@idiap.ch>
 *
 * Copyright (C) 2011-2012 Idiap Research Institute, Martigny, Switzerland
 * 
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, version 3 of the License.
 * 
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 * 
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

#include "bob/core/logging.h"
#include "bob/daq/VideoReaderCamera.h"

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
  bob::core::info << videoReader->info().c_str() << std::endl;
}

Camera::CamPixFormat VideoReaderCamera::getCamPixFormat() const {
  return Camera::RGB24;
}

void VideoReaderCamera::setCamPixFormat(Camera::CamPixFormat pixelFormat) {
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

int VideoReaderCamera::getSupportedCamPixFormats(std::vector<Camera::CamPixFormat>& pixelFormats) {
  pixelFormats.clear();;
  pixelFormats.push_back(getCamPixFormat());

  return 0;
}

int VideoReaderCamera::getSupportedFrameSizes(Camera::CamPixFormat pixelFormat, std::vector<FrameSize>& frameSizes) {
  frameSizes.clear();
  if (pixelFormat == getCamPixFormat()) {
    frameSizes.push_back(getFrameSize());
  }

  return 0;
}

int VideoReaderCamera::getSupportedFrameIntervals(Camera::CamPixFormat pixelFormat, FrameSize& frameSize,
                                         std::vector<FrameInterval>& frameIntervals) {
  frameIntervals.clear();
  if (pixelFormat == getCamPixFormat() && frameSize == getFrameSize()) {
    frameIntervals.push_back(getFrameInterval());
  }
  
  return 0;
}

}}
