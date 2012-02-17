/**
 * @file cxx/daq/daq/V4LCamera.h
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
