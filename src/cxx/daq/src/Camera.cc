#include "daq/Camera.h"

namespace bob { namespace daq {
  
static pthread_mutex_t pthread_mutex_initializer = PTHREAD_MUTEX_INITIALIZER;

Camera::Camera() : callbacks_mutex(pthread_mutex_initializer) {

}

Camera::~Camera() {

}

void Camera::addCameraCallback(CameraCallback& callback) {
  pthread_mutex_lock(&callbacks_mutex);
  callbacks.push_back(&callback);
  pthread_mutex_unlock(&callbacks_mutex);
}

void Camera::removeCameraCallback(CameraCallback& callback) {
  std::vector<CameraCallback*>::iterator it;

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