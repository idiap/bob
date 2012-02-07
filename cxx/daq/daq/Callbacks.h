/**
 * @file cxx/daq/daq/Callbacks.h
 * @date Thu Feb 2 11:22:57 2012 +0100
 * @author Francois Moulin <Francois.Moulin@idiap.ch>
 *
 * Copyright (C) 2011-2012 Idiap Reasearch Institute, Martigny, Switzerland
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
#ifndef CALLBACKS_H
#define CALLBACKS_H

#include <blitz/array.h>

namespace bob { namespace daq {

class Stoppable {
public:
  virtual void stop() = 0;
};


/**
 * Callback provided by a @c Controller
 */
class ControllerCallback : public Stoppable {
public:
  
  /**
   * Capture status
   */
  struct CaptureStatus {
    /// Is the controller recording the video? 
    bool isRecording;

    /// Total time of the capture session (in seconds)
    double totalSessionTime;
    
    /// Delay before recording (in seconds)
    double recordingDelay;

    /// Elapsed time (in seconds)
    double elapsedTime;

    /// Frame number
    int frameNb;
  };

  /**
   * Image received by the Controller.
   *
   * @param image pixel array in RGB 24 format
   * @param status information about the frame
   */
  virtual void imageReceived(blitz::Array<unsigned char, 2>& image, CaptureStatus& status) = 0;
};

class KeyPressCallback {
public:
  /**
   * @param key ASCII value of the pressed key
   */
  virtual void keyPressed(int key) = 0;
};

class FaceLocalizationCallback {
public:
  struct BoundingBox {
    bool detected;
    
    int x;
    int y;
    int width;
    int height;

    BoundingBox(int x, int y, int width, int height) : detected(true),
      x(x), y(y), width(width), height(height) {
      
    }

    BoundingBox(bool detected = false) : detected(detected) {
      
    }
  };

  virtual void onDetection(BoundingBox& bb) = 0;
};

}}
#endif // CALLBACKS_H