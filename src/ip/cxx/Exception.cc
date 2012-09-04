/**
 * @file cxx/ip/src/Exception.cc
 * @date Tue Mar 8 12:06:10 2011 +0100
 * @author Laurent El Shafey <Laurent.El-Shafey@idiap.ch>
 *
 * @brief Implementation of the exceptions used throughout the IP subsystem of
 * bob.
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

#include <boost/format.hpp>
#include "ip/Exception.h"

namespace ip = bob::ip;
namespace core = bob::core;

ip::Exception::Exception() throw() {
}

ip::Exception::~Exception() throw() {
}

const char* ip::Exception::what() const throw() {
  static const char* what_string = "Generic ip::Exception: user \
    specialization has not overwritten what() or is throwing an object of \
    this class (in which case, please fix it!)";
  return what_string;
}


ip::ParamOutOfBoundaryError::ParamOutOfBoundaryError(
  const std::string& paramname, const bool larger, const int value, 
  const int limit) throw(): 
    m_paramname(paramname), m_larger(larger), m_value(value), m_limit(limit) 
{
}

ip::ParamOutOfBoundaryError::~ParamOutOfBoundaryError() throw() {
}

const char* ip::ParamOutOfBoundaryError::what() const throw() {
  try {
    static const char* s_larger = "larger";
    static const char* s_smaller = "smaller";
    const char* s_selected = ( m_larger ? s_larger : s_smaller);
    boost::format message(
      "Parameter '%s' (value=%d) is %s than the limit %d.");
    message % m_paramname;
    message % m_value;
    message % s_selected;
    message % m_limit;
    m_message = message.str();
    return m_message.c_str();
  } catch (...) {
    static const char* emergency = "ip::ParamOutOfBoundaryError: cannot \
      format, exception raised";
    return emergency;
  }
}


ip::UnknownScalingAlgorithm::UnknownScalingAlgorithm() throw() {
}

ip::UnknownScalingAlgorithm::~UnknownScalingAlgorithm() throw() {
}

const char* ip::UnknownScalingAlgorithm::what() const throw() {
  static const char* what_string = "Generic ip::UnknownScalingAlgorithm:  \
    The given scaling algorithm is not valid!";
  return what_string;
}


ip::UnknownRotatingAlgorithm::UnknownRotatingAlgorithm() throw() {
}

ip::UnknownRotatingAlgorithm::~UnknownRotatingAlgorithm() throw() {
}

const char* ip::UnknownRotatingAlgorithm::what() const throw() {
  static const char* what_string = "Generic ip::UnknownRotatingAlgorithm:  \
    The given rotating algorithm is not valid!";
  return what_string;
}

ip::LBPRadiusDoesNotMatch::LBPRadiusDoesNotMatch(const std::string& axis,const std::string& plane1,const std::string& plane2) throw(): 
    m_axis(axis), m_plane1(plane1), m_plane2(plane2)
{
}

ip::LBPRadiusDoesNotMatch::~LBPRadiusDoesNotMatch() throw(){
}


const char* ip::LBPRadiusDoesNotMatch::what() const throw() {
   boost::format message(
   "The radius in '%s' direction does not match in planes %s and %s ");
   message % m_axis;
   message % m_plane1;
   message % m_plane2;
   m_message = message.str();
   return m_message.c_str();

}



ip::LBPUnsupportedNNeighbours::LBPUnsupportedNNeighbours(
  const int N) throw(): 
    m_n_neighbours(N)
{
}

ip::LBPUnsupportedNNeighbours::~LBPUnsupportedNNeighbours() throw() {
}

const char* ip::LBPUnsupportedNNeighbours::what() const throw() {
  try {
    boost::format message(
      "The LBP operator is not implemented for a number '%d' of neighbour \
        pixels. Only LBP4R and LBP8R are currently supported.");
    message % m_n_neighbours;
    m_message = message.str();
    return m_message.c_str();
  } catch (...) {
    static const char* emergency = "ip::LBPUnsupportedNNeighbours: cannot \
      format, exception raised";
    return emergency;
  }
}
