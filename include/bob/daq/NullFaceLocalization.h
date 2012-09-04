/**
 * @file cxx/daq/daq/NullFaceLocalization.h
 * @date Thu Feb 23 11:22:57 2012 +0100
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
#ifndef NULLFACERECOGNTION_H
#define NULLFACERECOGNTION_H

#include <daq/FaceLocalization.h>

namespace bob { namespace daq {

/**
 * @c NullFaceLocalization is an FaceLocalization which does nothing.
 */
class NullFaceLocalization : public FaceLocalization {
public:
  NullFaceLocalization() {}
  virtual ~NullFaceLocalization() {}

  bool start() {return true;}
  void stop() {};

  void imageReceived(blitz::Array<unsigned char, 2>& image, CaptureStatus& status) {}
};

}}

#endif // NULLFACERECOGNTION_H
