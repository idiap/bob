#ifndef FACERECOGNTION_H
#define FACERECOGNTION_H

#include <daq/Callbacks.h>
#include <vector>

namespace bob { namespace daq {

/**
 * @c FaceLocalization is an abstract class which provides face localization
 */
class FaceLocalization : public ControllerCallback {
public:
  FaceLocalization();
  virtual ~FaceLocalization();

  /**
   * Start the face localization of incoming frames
   */
  virtual bool start() = 0;
  
  void addFaceLocalizationCallback(FaceLocalizationCallback& callback);
  void removeFaceLocalizationCallback(FaceLocalizationCallback& callback);
  
protected:
  std::vector<FaceLocalizationCallback*> callbacks;
  pthread_mutex_t callbacks_mutex;
};

}}
#endif // FACERECOGNTION_H
