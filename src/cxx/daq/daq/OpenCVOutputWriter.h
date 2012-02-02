#ifndef OPENCVOUTPUTWRITER_H
#define OPENCVOUTPUTWRITER_H


#include <iostream>
#include <fstream>

#include "OutputWriter.h"

namespace bob { namespace daq {

/**
 * Write a video file using OpenCV.
 * 
 * Two files are created:
 * - .avi contains the video with a fixed fps
 * - .txt contains the timestamps for each frame
 */
class OpenCVOutputWriter:public OutputWriter {
public:
  OpenCVOutputWriter();
  virtual ~OpenCVOutputWriter();

  void writeFrame(blitz::Array<unsigned char, 2>& image, int frameNb, double timestamp);
  void open(int width, int height, int fps);
  void close();
  
private:
  cv::VideoWriter* videoWriter;
  std::ofstream* textFile;
};

}}
#endif // OPENCVOUTPUTWRITER_H
