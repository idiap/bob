/**
 * @file bob/daq/Controller.h
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
#ifndef CONTROLER_H
#define CONTROLER_H

#include "bob/daq/Camera.h"
#include "bob/daq/Callbacks.h"
#include "bob/daq/OutputWriter.h"

namespace bob { namespace daq {

/**
 * @c Controller is an abstract class which controls the capture process. It is
 * responsible to forward captured images to other classes, and have to convert
 * it to RGB24 format if needed.
 */
class Controller : public Camera::CameraCallback, public KeyPressCallback {
public:
  Controller();
  virtual ~Controller();
  
  void addControllerCallback(ControllerCallback& callback);
  void removeControllerCallback(ControllerCallback& callback);

  /**
   * Add classes that should be stopped in priority (i.e. before
   * @c ControllerCallback classes)
   */
  void addStoppable(Stoppable& stoppable);
  void removeStoppable(Stoppable& stoppable);
  
  /**
    * Get recording delay in seconds, i.e. amount of seconds before the
    * recording begins.
   */
  int getRecordingDelay();

  /// @see getRecordingDelay()
  void setRecordingDelay(int recordingDelay);

  /**
   * Get recording length in seconds (recording delay excluded)
   */
  int getLength();

  /// @see getLength()
  void setLength(int length);

  /**
   * Set the @c OutputWriter. Could be NULL.
   */
  void setOutputWriter(OutputWriter& outputWriter);
  
protected:
  std::vector<ControllerCallback*> callbacks;
  pthread_mutex_t callbacks_mutex;
  
  std::vector<Stoppable*> stoppables;
  pthread_mutex_t stoppables_mutex;

  OutputWriter* outputWriter;
  int length;
  int recordingDelay;
};

}}

#endif // CONTROLER_H
