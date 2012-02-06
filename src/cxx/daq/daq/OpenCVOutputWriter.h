/**
 * @file cxx/daq/daq/OpenCVOutputWriter.h
 * @date Thu Feb 2 11:22:57 2012 +0100
 * @author Francois Moulin <Francois.Moulin@idiap.ch>
 *
 * Copyright (C) 2011-2012 Idiap Reasearch Institute, Martigny, Switzerland
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
