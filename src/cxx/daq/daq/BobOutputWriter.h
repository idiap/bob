#ifndef BOBOUTPUTWRITER_H
#define BOBOUTPUTWRITER_H

#include <daq/OutputWriter.h>
#include <io/VideoWriter.h>
#include <iostream>
#include <fstream>

namespace bob { namespace daq {

/**
 * Write a video file using Bob.
 * 
 * Two files are created:
 * - .avi contains the video with a fixed fps
 * - .txt contains the timestamps for each frame
 */
class BobOutputWriter: public OutputWriter {
public:
  BobOutputWriter();
  virtual ~BobOutputWriter();
  
  void close();
  void open(int width, int height, int fps);
  void writeFrame(blitz::Array<unsigned char, 2>& image, int frameNb, double timestamp);

private:
  io::VideoWriter* videoWriter;
  std::ofstream* textFile;
};

}}

#endif // BOBOUTPUTWRITER_H
