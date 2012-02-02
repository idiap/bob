#ifndef CAPTURESYSTEM_H
#define CAPTURESYSTEM_H

#include <string>

#include <daq/Camera.h>
#include <boost/shared_ptr.hpp>

namespace bob { namespace daq {

/**
 * @c CaptureSystem is the main class used to capture images from a @c Camera
 * and save it in a video file. @c CaptureSystem also displays a GUI with
 * useful information about the current capture (remaining time, face detection,
 * ...)
 */
class CaptureSystem {
public:
  /**
   * Constructor
   *
   * @param camera properly initialized @c Camera (used to grab the images)
   * @param faceLocalizationModelPath path to the Visioner face localization model
   */
  CaptureSystem(boost::shared_ptr<Camera> camera, const char* faceLocalizationModelPath);
  virtual ~CaptureSystem();

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
   * Start the capture system. This call is blocking.
   */
  void start();

  /**
   * Get directory where captured images are written
   */
  std::string getOutputDir();

  /// @see getOutputDir()
  void setOutputDir(const std::string& dir);

  /**
   * Get output name
   */
  std::string getOutputName();

  /// @see getOutputName()
  void setOutputName(const std::string& name);


  /**
   * Get path to an image, displayed as thumbnail in the GUI
   */
  std::string getThumbnail();

  /// @see getThumbnail()
  void setThumbnail(const std::string& path);

  /**
   * Get whether GUI should be fullscreen
   */
  bool getFullScreen();

  /// @see getFullScreen()
  void setFullScreen(bool fullscreen);

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
  
private:
  boost::shared_ptr<Camera> camera;
  int recordingDelay;
  int length;

  std::string outputDir;
  std::string outputName;
  std::string thumbnail;
  std::string onStartRecording;
  std::string onStopRecording;

  bool fullscreen;
  int displayWidth;
  int displayHeight;

  std::string text;
  const char* faceLocalizationModelPath;
};

}}
#endif // CAPTURESYSTEM_H
