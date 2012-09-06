/**
 * @file daq/cxx/ConsoleDisplay.cc
 * @date Thu Feb 2 11:22:57 2012 +0100
 * @author Francois Moulin <Francois.Moulin@idiap.ch>
 *
 * Copyright (C) 2011-2012 Idiap Research Institute, Martigny, Switzerland
 * 
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, version 3 of the License.
 * 
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 * 
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */
#include "bob/daq/ConsoleDisplay.h"

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