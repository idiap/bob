#include "daq/ConsoleDisplay.h"

namespace bob { namespace daq {

static pthread_mutex_t pthread_mutex_initializer = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t pthread_cond_initializer = PTHREAD_COND_INITIALIZER;

ConsoleDisplay::ConsoleDisplay() : mutex(pthread_mutex_initializer), cond(pthread_cond_initializer) {
  mustStop = false;
}

ConsoleDisplay::~ConsoleDisplay() {

}

void ConsoleDisplay::stop() {
  pthread_mutex_lock(&mutex);
  mustStop = true;
  pthread_cond_signal(&cond);
  pthread_mutex_unlock(&mutex);
}

void ConsoleDisplay::start() {
  mustStop = false;

  // We need to wait until stop is called
  pthread_mutex_lock(&mutex);
  while(!mustStop) {
    pthread_cond_wait(&cond, &mutex);
  }
  
  mustStop = false;
  pthread_mutex_unlock(&mutex);
}

void ConsoleDisplay::imageReceived(blitz::Array<unsigned char, 2>& image, CaptureStatus& status) {
  printf("Frame %d received (%f sec) %s\n", status.frameNb, status.elapsedTime,
         (status.isRecording ? "recording" : "not recording"));
}

void ConsoleDisplay::onDetection (BoundingBox& bb) {
  printf("Face detect result: %s\n", (bb.detected ? "detected" : "not detected"));
}

}}