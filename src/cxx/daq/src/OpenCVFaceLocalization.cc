#include "daq/OpenCVFaceLocalization.h"

namespace bob { namespace daq {

static pthread_mutex_t pthread_mutex_initializer = PTHREAD_MUTEX_INITIALIZER;

OpenCVFaceLocalization::OpenCVFaceLocalization(const char* model_path) : mustStop(false), img_mutex(pthread_mutex_initializer) {
  // Load the HaarClassifierCascade
  cascade = (CvHaarClassifierCascade*)cvLoad(model_path, 0, 0, 0 );
  thread = 0;
}

OpenCVFaceLocalization::~OpenCVFaceLocalization() {
  stop();

  if (thread != 0) {
    // Ensure that the localization thread is over
    pthread_join(thread, NULL);
  }
  
  cvReleaseHaarClassifierCascade(&cascade);
}

void OpenCVFaceLocalization::stop() {
  mustStop = true;
}

static void* localize_(void* param) {
  OpenCVFaceLocalization* fl = (OpenCVFaceLocalization*) param;

  fl->localize();
}

// Detect faces
// see http://opencv.willowgarage.com/wiki/FaceDetection
FaceLocalizationCallback::BoundingBox detect_(cv::Mat* image, CvHaarClassifierCascade* cascade)
{
  // Create memory for calculations
  static CvMemStorage* storage = 0;
  
  // Create two points to represent the face locations
  CvPoint pt1, pt2;
  
  // Allocate the memory storage
  storage = cvCreateMemStorage(0);
  
  // Clear the memory storage which was used before
  cvClearMemStorage( storage );

  // There can be more than one face in an image. So create a growable sequence of faces.
  // Detect the objects and store them in the sequence
  CvSeq* faces = cvHaarDetectObjects(image, cascade, storage,
                                     1.1, 2, CV_HAAR_DO_CANNY_PRUNING,
                                     cvSize(40, 40) );

  FaceLocalizationCallback::BoundingBox bb;
  if (faces && faces->total > 0) {
    bb.detected = true;
    
    // Get the first face
    CvRect* r = (CvRect*)cvGetSeqElem( faces, 0 );

    bb.x = r->x;
    bb.y = r->y;
    bb.width = r->width;
    bb.height = r->height;
  }

  return bb;
}

void OpenCVFaceLocalization::localize() {
  // Main localization loop
  while(!mustStop) {
    cv::Mat gray;
    
    pthread_mutex_lock(&img_mutex);
    if (!img.empty()) {
      img.convertTo(gray, CV_8UC3);
    }
    pthread_mutex_unlock(&img_mutex);

    if(!gray.empty()) {
      FaceLocalizationCallback::BoundingBox bb = detect_(&gray, cascade);
      
      pthread_mutex_lock(&callbacks_mutex);
      for(std::vector<FaceLocalizationCallback*>::iterator it = callbacks.begin(); it != callbacks.end(); it++) {
        (*it)->onDetection(bb);
      }
      pthread_mutex_unlock(&callbacks_mutex);
    }
  }

  mustStop = false;
}

bool OpenCVFaceLocalization::start() {
  int error = pthread_create(&thread, NULL, localize_, (void*)this);

  if (error != 0) {
    thread = 0;
    return false;
  }

  return true;
}

void OpenCVFaceLocalization::imageReceived(blitz::Array<unsigned char, 2>& image, CaptureStatus& status) {
  pthread_mutex_lock(&img_mutex);
  img = cv::Mat(cv::Size(image.cols() / 3, image.rows()), CV_8UC3, image.data()).clone();
  pthread_mutex_unlock(&img_mutex);
}

}}