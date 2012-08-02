/**
 * @file cxx/daq/src/VisionerFaceLocalization.cc
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

#include "core/cast.h"
#include "daq/VisionerFaceLocalization.h"
#include "ip/color.h"
#include "ip/scale.h"
#include "io/Array.h"

namespace bob { namespace daq {

static pthread_mutex_t pthread_mutex_initializer = PTHREAD_MUTEX_INITIALIZER;

VisionerFaceLocalization::VisionerFaceLocalization(const char* model_path) : 
  img_mutex(pthread_mutex_initializer) {

  thread = 0;
  mustStop = false;
  
  bool ok = true;
  try {
    detector.reset(new bob::visioner::CVDetector(model_path));
    detector->m_type = bob::visioner::CVDetector::Scanning;
    detector->set_scan_levels(10);
  }
  catch(...) {
    ok = false;
  }

  if(!ok) {
    fprintf(stderr, "Error loading Visioner model %s\n", model_path);
    // FIXME Throw an exception
    exit(1);
  }
  
  lastid = -1;
  status.frameNb = -1;
}

VisionerFaceLocalization::~VisionerFaceLocalization() {
  stop();
  if (thread != 0) {
    pthread_join(thread, NULL);
  }
}

static void* localize_(void* param) {
  VisionerFaceLocalization* fl = (VisionerFaceLocalization*) param;

  fl->localize();
  return NULL;
}

void VisionerFaceLocalization::localize() {
  while(!mustStop) {
    // Downscale the image to be faster
    float downscale = 6;
    
    pthread_mutex_lock(&img_mutex);

    // Check that we don't do the localization on the same images
    if(lastid == status.frameNb) {
      pthread_mutex_unlock(&img_mutex);
      continue;
    }

    // Ensure that we don't have a too small image
    if (img.rows() < downscale * 60 || img.cols() < downscale * 60) {
      downscale = 2;
    }

    blitz::Array<bob::visioner::grey_t, 2> gray;
    if (img.size() != 0) {
      // Convert 2D to 3D blitz array
      blitz::Array<unsigned char, 3> image3D(img.data(), blitz::shape(img.rows(), img.cols() / 3, 3), blitz::neverDeleteData);
      // Reorder dimensions to be compatible with Bob
      blitz::Array<unsigned char, 3> imageBob(image3D.transpose(2, 0, 1));

      // Convert to grayscale
      blitz::Array<unsigned char, 2> grayUchar(img.rows(), img.cols() / 3);
      bob::ip::rgb_to_gray(imageBob, grayUchar);

      // Resize the image
      blitz::Array<double, 2> grayResized(grayUchar.rows()/downscale, grayUchar.cols()/downscale);
      bob::ip::scale(grayUchar, grayResized);

      // Convert to unsigned
      gray.resize(grayResized.shape());
      gray = bob::core::cast<bob::visioner::grey_t>(grayResized);
    }
    pthread_mutex_unlock(&img_mutex);

    if(gray.size() != 0) {
      visioner::detections_t detections;
      bool ok = detector->load(gray.data(), gray.rows(), gray.cols());

      if (!ok) {
        fprintf(stderr, "Visioner can't load image\n");
        continue;
      }

      // Detection
      detector->scan(detections);
      detector->sort_desc(detections);
  
      if (detections.size() == 0) {
        fprintf(stderr, "Visioner cannot find faces on image\n");
        continue;
      }

      visioner::detection_t& detect = detections[0];

      FaceLocalizationCallback::BoundingBox bb;

      qreal x, y, width, height;
      detect.second.first.getRect(&x, &y, &width, &height);
      if (width == 0 || height == 0) {
        bb.detected = false;
      }
      else {
        bb.detected = true;
        bb.x = (int)(x * downscale);
        bb.y = (int)(y * downscale);
        bb.width = (int)(width * downscale);
        bb.height = (int)(height * downscale);
      }
      
      pthread_mutex_lock(&callbacks_mutex);
      for(std::vector<FaceLocalizationCallback*>::iterator it = callbacks.begin(); it != callbacks.end(); it++) {
        (*it)->onDetection(bb);
      }
      pthread_mutex_unlock(&callbacks_mutex);
      
    }
  }

  mustStop = false;
}

void VisionerFaceLocalization::stop() {
  mustStop = true;
}

bool VisionerFaceLocalization::start() {
  int error = pthread_create(&thread, NULL, localize_, (void*)this);

  if (error != 0) {
    thread = 0;
    return false;
  }

  return true;
}

void VisionerFaceLocalization::imageReceived(blitz::Array<unsigned char, 2>& image, CaptureStatus& status) {
  pthread_mutex_lock(&img_mutex);
  this->img.resize(image.shape());
  this->img = image;
  this->status = status;
  pthread_mutex_unlock(&img_mutex);
}

}}
