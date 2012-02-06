#include "daq/VisionerFaceLocalization.h"
#include <visioner/proc/detection.h>
#include <ip/color.h>
#include <ip/scale.h>
#include <io/Array.h>

namespace bob { namespace daq {

/**
 * Private member structure
 */
struct Visioner_ptr {
  visioner::Model* model;
  visioner::SWScanner* scanner;

  ~Visioner_ptr() {
    delete model;
    delete scanner;
  }
};

static pthread_mutex_t pthread_mutex_initializer = PTHREAD_MUTEX_INITIALIZER;

VisionerFaceLocalization::VisionerFaceLocalization(const char* model_path) : img_mutex(pthread_mutex_initializer) {
  thread = 0;
  mustStop = false;
  visioner::param_t param;
  visioner_ptr = new Visioner_ptr;
  
  visioner_ptr->model = new visioner::Model;
  
  bool ok = false;
  try {
    ok = visioner::load_model(param, *visioner_ptr->model, model_path);
  }
  catch(...) {
    ok = false;
  }

  if(!ok || visioner_ptr->model->size() != 1) {
    fprintf(stderr, "Error loading Visioner model %s\n", model_path);
    // FIXME Throw an exception
    exit(1);
  }

  
  visioner_ptr->scanner = new visioner::SWScanner(param);
  lastid = -1;
  status.frameNb = -1;
}

VisionerFaceLocalization::~VisionerFaceLocalization() {
  stop();
  if (thread != 0) {
    pthread_join(thread, NULL);
  }
  delete visioner_ptr;
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

    blitz::Array<short, 2> gray;
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

      // Convert to short
      gray.resize(grayResized.shape());
      gray = bob::core::cast<short>(grayResized);
    }
    pthread_mutex_unlock(&img_mutex);

    if(gray.size() != 0) {
      visioner::detection_t detect;
      visioner::grey_t* image = gray.data();

      bool ok = visioner_ptr->scanner->load((visioner::grey_t*)image, gray.rows(), gray.cols());

      if (!ok) {
        fprintf(stderr, "Visioner can't load image\n");
        continue;
      }

      // Detection
      visioner::detect_max(visioner_ptr->model->at(0), 0, *visioner_ptr->scanner, detect);

      FaceLocalizationCallback::BoundingBox bb;

      qreal x, y, width, height;
      detect.second.getRect(&x, &y, &width, &height);
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