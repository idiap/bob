#ifndef CONSOLEDISPLAY_H
#define CONSOLEDISPLAY_H

#include <daq/Display.h>

namespace bob { namespace daq {

/**
 * Dispay class that prints a console message when a frame or a detection is
 * received.
 */
class ConsoleDisplay : public bob::daq::Display {
public:
  ConsoleDisplay();
  virtual ~ConsoleDisplay();
  
  void stop();
  void start();
  void imageReceived(blitz::Array<unsigned char, 2>& image, CaptureStatus& status);
  void onDetection(BoundingBox& bb);

private:
  pthread_mutex_t mutex;
  pthread_cond_t cond;

  bool mustStop;
};

}}

#endif // CONSOLEDISPLAY_H
