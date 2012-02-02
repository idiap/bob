#include "daq/Display.h"

namespace bob { namespace daq {
  
static pthread_mutex_t pthread_mutex_initializer = PTHREAD_MUTEX_INITIALIZER;

Display::Display() : callbacks_mutex(pthread_mutex_initializer) {
  fullscreen = false;
  displayWidth = -1;
  displayHeight = -1;
}

Display::~Display() {

}

void Display::addKeyPressCallback(KeyPressCallback& callback) {
  pthread_mutex_lock(&callbacks_mutex);
  callbacks.push_back(&callback);
  pthread_mutex_unlock(&callbacks_mutex);
}

void Display::removeKeyPressCallback(KeyPressCallback& callback) {
  std::vector<KeyPressCallback*>::iterator it;

  pthread_mutex_lock(&callbacks_mutex);
  for(it = callbacks.begin(); it != callbacks.end(); it++) {
    if ((*it) == &callback) {
      callbacks.erase(it);
      break;
    }
  }
  pthread_mutex_unlock(&callbacks_mutex);
}

void Display::setThumbnail(std::string& path) {
  this->thumbnail = path;
}

void Display::setFullscreen(bool fullscreen) {
  this->fullscreen = fullscreen;
}

void Display::setDisplaySize(int width, int height) {
  this->displayWidth = width;
  this->displayHeight = height;
}


void Display::setExecuteOnStartRecording(const std::string& program) {
  onStartRecording = program;
}

void Display::setExecuteOnStopRecording(const std::string& program) {
  onStopRecording = program;
}

void Display::setText(const std::string& text) {
  this->text = text;
}

}}