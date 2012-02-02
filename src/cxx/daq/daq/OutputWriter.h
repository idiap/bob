#ifndef OUTPUTWRITER_H
#define OUTPUTWRITER_H

#include <string>
#include <blitz/array.h>

namespace bob { namespace daq {
  
/**
 * @c OutputWriter is an abstract class which provides a way to write frames on
 * the hard drive
 */
class OutputWriter {
public:
  OutputWriter();
  virtual ~OutputWriter();

  /**
   * Write a frame on the hard drive.
   *
   * @param image     pixels in RGB24 format
   * @param frameNb   frame number
   * @param timestamp frame timestamp in seconds
   */
  virtual void writeFrame(blitz::Array<unsigned char, 2>& image, int frameNb, double timestamp) = 0;

  virtual void open(int width, int height, int fps) = 0;
  virtual void close() = 0;

  /**
   * Set the directory where we want to output
   */
  void setOutputDir(std::string dir);

  /**
   * Set the name used to identify the output files
   */
  void setOutputName(std::string name);
  
protected:
  int width;
  int height;

  std::string dir;
  std::string name;
};

}}

#endif // OUTPUTWRITER_H
