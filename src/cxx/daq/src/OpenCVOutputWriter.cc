#include "daq/OpenCVOutputWriter.h"

namespace bob { namespace daq {

OpenCVOutputWriter::OpenCVOutputWriter() {
  videoWriter = NULL;
  textFile = NULL;
}

OpenCVOutputWriter::~OpenCVOutputWriter() {
  delete videoWriter;
  delete textFile;
}

void OpenCVOutputWriter::writeFrame(blitz::Array<unsigned char, 2>& image, int frameNb, double timestamp) {
  cv::Mat img = cv::Mat(cv::Size(image.cols() / 3, image.rows()), CV_8UC3, image.data());
  if (videoWriter != NULL) {
    *videoWriter << img;
    *textFile << frameNb << " " << timestamp << std::endl;
  }
}

void OpenCVOutputWriter::open(int width, int height, int fps) {
  if (videoWriter != NULL) {
    close();
  }
  else {
    videoWriter = new cv::VideoWriter(dir + "/" + name + ".avi", CV_FOURCC('P','I','M','1'), fps, cv::Size(width, height), true);
    textFile = new std::ofstream((dir + "/" + name + ".txt").c_str(), std::ios::out);
  }
}

void OpenCVOutputWriter::close() {
  delete videoWriter;
  delete textFile;
  videoWriter = NULL;
  textFile = NULL;
}

}}