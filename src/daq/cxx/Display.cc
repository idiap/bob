/**
 * @file daq/cxx/Display.cc
 * @date Thu Feb 2 11:22:57 2012 +0100
 * @author Francois Moulin <Francois.Moulin@idiap.ch>
 *
 * Copyright (C) 2011-2013 Idiap Research Institute, Martigny, Switzerland
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
#include "bob/daq/Display.h"

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

