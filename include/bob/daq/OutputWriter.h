/**
 * @file cxx/daq/daq/OutputWriter.h
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
