/**
 * @file daq/cxx/BobOutputWriter.cc
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

#include <boost/make_shared.hpp>

#include "bob/daq/BobOutputWriter.h"
#include "bob/core/blitz_array.h"

namespace bob { namespace daq {
  
BobOutputWriter::BobOutputWriter() {
}

BobOutputWriter::~BobOutputWriter() {
}

void BobOutputWriter::close() {
  if (videoWriter) {
    videoWriter->close();
    textFile->close();
    videoWriter.reset();
    textFile.reset();
  }
}

void BobOutputWriter::open(int width, int height, int fps) {
  if (videoWriter) close();
  
  videoWriter = boost::make_shared<io::VideoWriter>(dir + "/" + name + ".avi", height, width, fps);
  textFile = boost::make_shared<std::ofstream>((dir + "/" + name + ".txt").c_str(), std::ios::out);
}

void BobOutputWriter::writeFrame(blitz::Array<unsigned char, 2>& image, int frameNb, double timestamp) {
  if (videoWriter) {
    // Convert image in Bob format (3D array)
    blitz::Array<unsigned char, 3> image3(image.data(), blitz::shape(image.rows(), image.cols() / 3, 3), blitz::neverDeleteData);

    core::array::blitz_array array(image3.transpose(2, 0, 1));
    videoWriter->append(array);
    *textFile << frameNb << " " << timestamp << std::endl;
  }
}

}}
