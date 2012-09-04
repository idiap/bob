/**
 * @file cxx/daq/daq/VisionerFaceLocalization.h
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
#ifndef VISIONERFACELOCALIZATION_H
#define VISIONERFACELOCALIZATION_H

#include <boost/shared_ptr.hpp>

#include "bob/daq/FaceLocalization.h"
#include "bob/visioner/cv/cv_detector.h"

namespace bob { namespace daq {

/**
 * Provide face localization using Visioner
 */
class VisionerFaceLocalization : public FaceLocalization {

public:
  /**
   * @param model_path path to a model file (e.g. Face.MCT9.gz)
   */
  VisionerFaceLocalization(const char* model_path);
  virtual ~VisionerFaceLocalization();
  
  void imageReceived(blitz::Array<unsigned char, 2>& image, CaptureStatus& status);

  void stop();
  bool start();

  
  void localize();
private:
  pthread_t thread;
  
  boost::shared_ptr<bob::visioner::CVDetector> detector;
  
  blitz::Array<unsigned char, 2> img;
  pthread_mutex_t img_mutex;
  CaptureStatus status;

  int lastid;

  bool mustStop;
};

}}
#endif // VISIONERFACELOCALIZATION_H
