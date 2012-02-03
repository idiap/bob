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