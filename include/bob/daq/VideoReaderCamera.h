/**
 * @file bob/daq/VideoReaderCamera.h
 * @date Thu Feb 2 11:22:57 2012 +0100
 * @author Francois Moulin <Francois.Moulin@idiap.ch>
 *
 * Copyright (C) 2011-2013 Idiap Research Institute, Martigny, Switzerland
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
#ifndef VIDEOREADERCAMERA_H
#define VIDEOREADERCAMERA_H

#include "bob/daq/Camera.h"
#include "bob/io/VideoReader.h"

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
