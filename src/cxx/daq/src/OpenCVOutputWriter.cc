/**
 * @file cxx/daq/src/OpenCVOutputWriter.cc
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