#include "daq/OpenCVDisplay.h"

#include <highgui.h>

namespace bob { namespace daq {

static pthread_mutex_t pthread_mutex_initializer = PTHREAD_MUTEX_INITIALIZER;

OpenCVDisplay::OpenCVDisplay() : image_mutex(pthread_mutex_initializer),
  boundingBox_mutex(pthread_mutex_initializer), mustStop(false) {
  captureStatus.isRecording = false;
  
  captureStatus.totalSessionTime = 0;
  captureStatus.recordingDelay = 0;
  captureStatus.elapsedTime = 0;

  fps = 0;
  fps_nbFrame = 0;
  fps_startTime = 0;

  fullscreen = false;
  displayWidth = -1;
  displayHeight = -1;
}

OpenCVDisplay::~OpenCVDisplay() {

}

void OpenCVDisplay::stop() {
  mustStop = true;
}

void OpenCVDisplay::start() {
  pthread_mutex_lock(&image_mutex);
  // Initialize image with a dummy image
  image = cv::Mat(200, 200, CV_8UC1);
  pthread_mutex_unlock(&image_mutex);

  // Contains a copy of image to work with
  cv::Mat working_image;
  // Contains a copy of status to work with
  CaptureStatus working_status;

  // This call fixes some problems on Linux
  cvStartWindowThread();

  if (fullscreen) {
    cv::namedWindow("OpenCVDisplay", CV_WINDOW_KEEPRATIO | CV_GUI_EXPANDED);
    cv::setWindowProperty("OpenCVDisplay", CV_WND_PROP_FULLSCREEN, CV_WINDOW_FULLSCREEN);
  }
  else if (displayHeight > 0 && displayWidth > 0) {
    cv::namedWindow("OpenCVDisplay", CV_WINDOW_KEEPRATIO | CV_GUI_EXPANDED);
    cvResizeWindow("OpenCVDisplay", displayWidth, displayHeight);
  }
  else {
    cv::namedWindow("OpenCVDisplay");
  }

  cv::Scalar blue(255, 0, 0);
  cv::Scalar red(0, 0, 255);

  // Contains the loaded thumbnail
  cv::Mat thumbnailImage;
  if (!thumbnail.empty()) {
    // Fix the maximum width or height
    int max_size = 100;
    
    thumbnailImage = cv::imread(thumbnail);
    
    int thumb_width = thumbnailImage.cols;
    int thumb_height = thumbnailImage.rows;

    // Compute the requested size
    if (thumb_width > thumb_height) {
      thumb_height = (thumb_height * max_size) / thumb_width;
      thumb_width = max_size;
    }
    else {
      thumb_width = (thumb_width * max_size) / thumb_height;
      thumb_height = max_size;
    }

    // Resize the thumbnail
    cv::resize(thumbnailImage, thumbnailImage, cv::Size(thumb_width, thumb_height));
  }

  pthread_mutex_lock(&boundingBox_mutex);
  boundingBox.detected = false;
  pthread_mutex_unlock(&boundingBox_mutex);

  // Store if the capture system was recording at the last frame
  bool was_recording = false;

  // Store if we have seen a captured frame already
  bool init_frame = true;

  // Main display loop
  while(!mustStop) {
    // Get pressed key and handle events
    int key = cv::waitKey(5);

    // Forward key events
    if (key != -1) {
      pthread_mutex_lock(&callbacks_mutex);
      for(std::vector<KeyPressCallback*>::iterator it = callbacks.begin(); it != callbacks.end(); it++) {
        (*it)->keyPressed(key);
      }
      pthread_mutex_unlock(&callbacks_mutex);
    }

    // Copy image and status to limit the critical section
    pthread_mutex_lock(&image_mutex);
    working_image = image.clone();
    working_status = captureStatus;
    pthread_mutex_unlock(&image_mutex);

    // Handle the start/stop recording events
    if (was_recording != working_status.isRecording) {
      was_recording = working_status.isRecording;

      if (working_status.isRecording) {
        if (!onStartRecording.empty()) {
          system(onStartRecording.c_str());
        }
      }
    }

    // Update init_frame
    if (init_frame && working_status.totalSessionTime > 0) {
      init_frame = false;
    }
    
    // Display thumbnail
    if (!init_frame && !thumbnailImage.empty()) {
      // Check the size of the thumbnail is not greater than image / 2
      if (thumbnailImage.cols > thumbnailImage.rows &&  thumbnailImage.cols > working_image.cols / 2 +1) {
        int max_size = working_image.cols / 2;
        
        int thumb_width = thumbnailImage.cols;
        int thumb_height = thumbnailImage.rows;
        
        thumb_height = (thumb_height * max_size) / thumb_width;
        thumb_width = max_size;

        cv::resize(thumbnailImage, thumbnailImage, cv::Size(thumb_width, thumb_height));
        
      }
      else if (thumbnailImage.cols <= thumbnailImage.rows &&  thumbnailImage.rows > working_image.rows / 2 +1) {
        int max_size = working_image.rows / 2;
        
        int thumb_width = thumbnailImage.cols;
        int thumb_height = thumbnailImage.rows;
        
        thumb_width = (thumb_width * max_size) / thumb_height;
        thumb_height = max_size;

        cv::resize(thumbnailImage, thumbnailImage, cv::Size(thumb_width, thumb_height));
      }

      // Draw the thumbnail on the current image
      cv::Mat roi = working_image(cv::Rect(0, 0, std::min(thumbnailImage.cols, working_image.cols),
                                         std::min(thumbnailImage.rows, working_image.rows)));
      thumbnailImage.copyTo(roi);
    }
    
    // Draw detection results
    pthread_mutex_lock(&boundingBox_mutex);
    if (boundingBox.detected) {
      cv::rectangle(working_image, cv::Rect(boundingBox.x, boundingBox.y, boundingBox.width, boundingBox.height), blue);
    }
    pthread_mutex_unlock(&boundingBox_mutex);

    // Compute useful times
    double delay_remaining = working_status.recordingDelay - working_status.elapsedTime;
    double total_remaining = working_status.totalSessionTime - working_status.elapsedTime;

    // Things to do before recording
    if (delay_remaining > 0) {
      /*
      std::stringstream ss;
      ss << "Waiting " << (int)delay_remaining+1 << " sec";
      int fontFace = cv::FONT_HERSHEY_PLAIN;
      double fontScale = 2;
      int thickness = 1;
      int baseline=0;
      
      cv::Size textSize = cv::getTextSize(ss.str(), fontFace, fontScale, thickness, &baseline);
      
      cv::putText(working_image, ss.str(), cv::Point(0, textSize.height + 10), fontFace, fontScale, red, thickness);
      */
    }

    // Things to do when recording
    if (total_remaining > 0 && working_status.totalSessionTime > 0) {

      // Display progress bar
      double fract = (working_status.totalSessionTime  - total_remaining) / working_status.totalSessionTime;
      cv::Rect rect(working_image.cols * fract, working_image.rows - 10, working_image.cols - working_image.cols * fract , 10);
      cv::rectangle(working_image, rect, blue, -1);

      if (delay_remaining > 0) {
        double fract_delay = 1 - (working_status.totalSessionTime  - working_status.recordingDelay) / working_status.totalSessionTime;
        cv::Rect rect(cv::Point(working_image.cols * fract, working_image.rows - 10),
                      cv::Point(working_image.cols * fract_delay , working_image.rows));
        cv::rectangle(working_image, rect, red, -1);
      }

      /*
      std::stringstream ss;
      ss << "" << (int)total_remaining+1 << " sec remaining";
      int fontFace = cv::FONT_HERSHEY_PLAIN;
      double fontScale = 1;
      int thickness = 1;
      int baseline=0;
      
      //cv::Size textSize = cv::getTextSize(ss.str(), fontFace, fontScale, thickness, &baseline);
      
      cv::putText(working_image, ss.str(), cv::Point(0, working_image.rows - 10), fontFace, fontScale, blue, thickness);
      */
    }

    /*
    // Display fps
    {
      std::stringstream ss;
      ss << "" << fps << " fps";
      int fontFace = cv::FONT_HERSHEY_PLAIN;
      double fontScale = 1;
      int thickness = 1;
      int baseline=0;
      
      //cv::Size textSize = cv::getTextSize(ss.str(), fontFace, fontScale, thickness, &baseline);
      
      cv::putText(working_image, ss.str(), cv::Point(working_image.cols / 2, 25), fontFace, fontScale, blue, thickness);
    }
    */
    
    // Display custom text
    {
      std::stringstream ss;
      ss << text;
      int fontFace = cv::FONT_HERSHEY_PLAIN;
      double fontScale = 2;
      int thickness = 1;
      int baseline = 0;
      
      cv::Size textSize = cv::getTextSize(ss.str(), fontFace, fontScale, thickness, &baseline);
      
      cv::putText(working_image, ss.str(), cv::Point(working_image.cols / 2 - textSize.width / 2, 25), fontFace, fontScale, red, thickness);
    }

    // Display recording circle
    if (working_status.isRecording) {
      cv::circle(working_image, cv::Point(working_image.cols - 20, 20), 5, red, -1);
    }
    else {
      cv::circle(working_image, cv::Point(working_image.cols - 20, 20), 5, blue, -1);
    }

    cv::imshow("OpenCVDisplay", working_image);
  }

  if (!onStopRecording.empty()) {
    system(onStopRecording.c_str());
  }

  cvDestroyWindow("OpenCVDisplay");
  cv::waitKey(5);
  mustStop = false;
}

void OpenCVDisplay::imageReceived(blitz::Array<unsigned char, 2>& image, CaptureStatus& status) {
  pthread_mutex_lock(&image_mutex);
  // Copy the image
  this->image = cv::Mat(cv::Size(image.cols() / 3, image.rows()), CV_8UC3, image.data()).clone();
  this->captureStatus = status;

  // Update fps information
  fps_nbFrame++;
  if (status.elapsedTime - fps_startTime > 1) {
    fps = fps_nbFrame / (status.elapsedTime - fps_startTime);
    fps_nbFrame = 0;
    fps_startTime = status.elapsedTime;
  }
  
  pthread_mutex_unlock(&image_mutex);
}


void OpenCVDisplay::onDetection(FaceLocalizationCallback::BoundingBox& bb) {
  pthread_mutex_lock(&boundingBox_mutex);
  // Copy bounding box
  boundingBox = bb;
  pthread_mutex_unlock(&boundingBox_mutex);
}

}}