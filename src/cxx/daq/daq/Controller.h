#ifndef CONTROLER_H
#define CONTROLER_H

#include <daq/Camera.h>
#include <daq/Callbacks.h>
#include "daq/OutputWriter.h"

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
