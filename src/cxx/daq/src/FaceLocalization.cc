#include "daq/FaceLocalization.h"

namespace bob { namespace daq {

static pthread_mutex_t pthread_mutex_initializer = PTHREAD_MUTEX_INITIALIZER;

FaceLocalization::FaceLocalization() :  callbacks_mutex(pthread_mutex_initializer) {
}

FaceLocalization::~FaceLocalization() {
}



void FaceLocalization::addFaceLocalizationCallback(FaceLocalizationCallback& callback) {
  pthread_mutex_lock(&callbacks_mutex);
  callbacks.push_back(&callback);
  pthread_mutex_unlock(&callbacks_mutex);
}

void FaceLocalization::removeFaceLocalizationCallback(FaceLocalizationCallback& callback) {
  std::vector<FaceLocalizationCallback*>::iterator it;

  pthread_mutex_lock(&callbacks_mutex);
  for(it = callbacks.begin(); it != callbacks.end(); it++) {
    if ((*it) == &callback) {
      callbacks.erase(it);
      break;
    }
  }
  pthread_mutex_unlock(&callbacks_mutex);
}

}}