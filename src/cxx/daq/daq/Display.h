#ifndef DISPLAY_H
#define DISPLAY_H

#include <daq/Callbacks.h>
#include <vector>

namespace bob { namespace daq {

/**
 * @c Display is an abstract class which is responsible to display an interface
 * to the user.
 */
class Display : public ControllerCallback, public FaceLocalizationCallback {
public:
  Display();
  virtual ~Display();

  /**
   * Start the interface. This call should be blocking.
   */
  virtual void start() = 0;

  /**
   * Stop the interface
   */
  virtual void stop() = 0;

  /**
   * Add a callback which listen to user keyboard interactions.
   */
  void addKeyPressCallback(KeyPressCallback& callback);

  /// @see addKeyPressCallback
  void removeKeyPressCallback(KeyPressCallback& callback);

  /**
   * Set path to an image, displayed as thumbnail in the GUI
   */
  void setThumbnail(std::string& path);
  
  /**
   * Set whether GUI should be fullscreen
   */
  void setFullscreen(bool fullscreen);
  
  /**
   * Set GUI size (ignored if fullscreen)
   */
  void setDisplaySize(int width, int height);

  /**
   * Set a shell command executed when the recording starts.
   * @warning The command blocks the GUI thread. You should execute time
   * consuming commands in a sub-shell (e.g. "command params &")
   */
  void setExecuteOnStartRecording(const std::string& program);
  
  /**
   * Set a shell command executed when the recording stops.
   * @warning See setExecuteOnStartRecording()
   */
  void setExecuteOnStopRecording(const std::string& program);

  /**
   * Set custom text displayed in the GUI
   */
  void setText(const std::string& text);
  
protected:
  std::vector<KeyPressCallback*> callbacks;
  pthread_mutex_t callbacks_mutex;

  std::string thumbnail;
  bool fullscreen;
  int displayWidth;
  int displayHeight;

  std::string onStartRecording;
  std::string onStopRecording;

  std::string text;
};

}}
#endif // DISPLAY_H
