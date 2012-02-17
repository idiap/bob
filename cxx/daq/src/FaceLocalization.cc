/**
 * @file cxx/daq/src/FaceLocalization.cc
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