/**
 * @file ip/cxx/Exception.cc
 * @date Tue Mar 8 12:06:10 2011 +0100
 * @author Laurent El Shafey <Laurent.El-Shafey@idiap.ch>
 *
 * @brief Implementation of the exceptions used throughout the IP subsystem of
 * bob.
 *
 * Copyright (C) 2011-2013 Idiap Research Institute, Martigny, Switzerland
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

#include <boost/format.hpp>
#include "bob/ip/Exception.h"

bob::ip::Exception::Exception() throw() {
}

bob::ip::Exception::~Exception() throw() {
}

const char* bob::ip::Exception::what() const throw() {
  static const char* what_string = "Generic bob::ip::Exception: user \
    specialization has not overwritten what() or is throwing an object of \
    this class (in which case, please fix it!)";
  return what_string;
}


bob::ip::UnknownScalingAlgorithm::UnknownScalingAlgorithm() throw() {
}

bob::ip::UnknownScalingAlgorithm::~UnknownScalingAlgorithm() throw() {
}

const char* bob::ip::UnknownScalingAlgorithm::what() const throw() {
  static const char* what_string = "Generic bob::ip::UnknownScalingAlgorithm:  \
    The given scaling algorithm is not valid!";
  return what_string;
}


bob::ip::UnknownRotatingAlgorithm::UnknownRotatingAlgorithm() throw() {
}

bob::ip::UnknownRotatingAlgorithm::~UnknownRotatingAlgorithm() throw() {
}

const char* bob::ip::UnknownRotatingAlgorithm::what() const throw() {
  static const char* what_string = "Generic bob::ip::UnknownRotatingAlgorithm:  \
    The given rotating algorithm is not valid!";
  return what_string;
}

bob::ip::LBPRadiusDoesNotMatch::LBPRadiusDoesNotMatch(const std::string& axis,const std::string& plane1,const std::string& plane2) throw(): 
    m_axis(axis), m_plane1(plane1), m_plane2(plane2)
{
}

bob::ip::LBPRadiusDoesNotMatch::~LBPRadiusDoesNotMatch() throw(){
}


const char* bob::ip::LBPRadiusDoesNotMatch::what() const throw() {
   boost::format message(
   "The radius in '%s' direction does not match in planes %s and %s ");
   message % m_axis;
   message % m_plane1;
   message % m_plane2;
   m_message = message.str();
   return m_message.c_str();

}



bob::ip::LBPUnsupportedNNeighbours::LBPUnsupportedNNeighbours(
  const int N) throw(): 
    m_n_neighbours(N)
{
}

bob::ip::LBPUnsupportedNNeighbours::~LBPUnsupportedNNeighbours() throw() {
}

const char* bob::ip::LBPUnsupportedNNeighbours::what() const throw() {
  try {
    boost::format message(
      "The LBP operator is not implemented for a number '%d' of neighbour \
        pixels. Only LBP4R and LBP8R are currently supported.");
    message % m_n_neighbours;
    m_message = message.str();
    return m_message.c_str();
  } catch (...) {
    static const char* emergency = "bob::ip::LBPUnsupportedNNeighbours: cannot \
      format, exception raised";
    return emergency;
  }
}
