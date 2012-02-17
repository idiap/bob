/**
 * @file cxx/daq/src/Controller.cc
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
#include "daq/Controller.h"

namespace bob { namespace daq {
  
static pthread_mutex_t pthread_mutex_initializer = PTHREAD_MUTEX_INITIALIZER;

Controller::Controller() : callbacks_mutex(pthread_mutex_initializer), stoppables_mutex(pthread_mutex_initializer) {
  recordingDelay = 3;
  length = 10;
  outputWriter = NULL;
}

Controller::~Controller() {
}



void Controller::addControllerCallback(ControllerCallback& callback) {
  pthread_mutex_lock(&callbacks_mutex);
  callbacks.push_back(&callback);
  pthread_mutex_unlock(&callbacks_mutex);
}

void Controller::removeControllerCallback(ControllerCallback& callback) {
  std::vector<ControllerCallback*>::iterator it;
  
  pthread_mutex_lock(&callbacks_mutex);
  for(it = callbacks.begin(); it != callbacks.end(); it++) {
    if ((*it) == &callback) {
      it = callbacks.erase(it);
      if (it == callbacks.end()) {
        break;
      }
    }
  }
  pthread_mutex_unlock(&callbacks_mutex);
}

void Controller::addStoppable(Stoppable& stoppable) {
  pthread_mutex_lock(&stoppables_mutex);
  stoppables.push_back(&stoppable);
  pthread_mutex_unlock(&stoppables_mutex);
}

void Controller::removeStoppable(Stoppable& stoppable) {
  std::vector<Stoppable*>::iterator it;
  
  pthread_mutex_lock(&stoppables_mutex);
  for(it = stoppables.begin(); it != stoppables.end(); it++) {
    if ((*it) == &stoppable) {
      it = stoppables.erase(it);
      if (it == stoppables.end()) {
        break;
      }
    }
  }
  pthread_mutex_unlock(&stoppables_mutex);
}

void Controller::setRecordingDelay(int recordingDelay) {
  this->recordingDelay = recordingDelay;
}

int Controller::getRecordingDelay() {
  return this->recordingDelay;
}

void Controller::setLength(int length) {
  this->length = length;
}

int Controller::getLength() {
  return this->length;
}

void Controller::setOutputWriter(OutputWriter& outputWriter) {
  this->outputWriter = &outputWriter;
}

}}