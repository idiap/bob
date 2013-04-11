/**
 * @file bob/ip/Exception.h
 * @date Tue Mar 8 12:06:10 2011 +0100
 * @author Laurent El Shafey <Laurent.El-Shafey@idiap.ch>
 *
 * @brief Exceptions used throughout the IP subsystem of bob
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

#ifndef BOB_IP_EXCEPTION_H
#define BOB_IP_EXCEPTION_H

#include <cstdlib>
#include "bob/core/Exception.h"

namespace bob { namespace ip {

  class Exception: public bob::core::Exception {

    public:
      Exception() throw();
      virtual ~Exception() throw();
      virtual const char* what() const throw();

  };

  class UnknownScalingAlgorithm: public Exception {
    public:
      UnknownScalingAlgorithm() throw();
      virtual ~UnknownScalingAlgorithm() throw();
      virtual const char* what() const throw();

    private:
      mutable std::string m_message;
  };

  class UnknownRotatingAlgorithm: public Exception {
    public:
      UnknownRotatingAlgorithm() throw();
      virtual ~UnknownRotatingAlgorithm() throw();
      virtual const char* what() const throw();

    private:
      mutable std::string m_message;
  };


  /**
   * This exception is thrown when the radius between LBPs does not match. For example, if the radius in X direction in XY plane is different from the radius in X direction in XT plane, the exception will raised.
   */
  class LBPRadiusDoesNotMatch: public Exception {
    public:
      LBPRadiusDoesNotMatch(const std::string& axis,const std::string& plane1,const std::string& plane2) throw();
      virtual ~LBPRadiusDoesNotMatch() throw();
      virtual const char* what() const throw();

    private:
      std::string m_axis;
      std::string m_plane1;
      std::string m_plane2;

  };


  /**
    * This exception is thrown when using a LBP with a non-common number
    * of neighbours ( != 4 && != 8)
   */
  class LBPUnsupportedNNeighbours: public Exception {
    public:
      LBPUnsupportedNNeighbours(const int N)  throw();
      virtual ~LBPUnsupportedNNeighbours() throw();
      virtual const char* what() const throw();

    private:
      int m_n_neighbours;
      mutable std::string m_message;
  };

}}

#endif /* BOB_IP_EXCEPTION_H */
