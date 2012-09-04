/**
 * @file cxx/daq/daq/SimpleController.h
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
#ifndef SIMPLECONTROLER_H
#define SIMPLECONTROLER_H

#include "bob/daq/Controller.h"

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
