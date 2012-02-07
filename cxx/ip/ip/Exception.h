/**
 * @file cxx/ip/ip/Exception.h
 * @date Tue Mar 8 12:06:10 2011 +0100
 * @author Laurent El Shafey <Laurent.El-Shafey@idiap.ch>
 *
 * @brief Exceptions used throughout the IP subsystem of bob
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

#ifndef BOB_IP_EXCEPTION_H 
#define BOB_IP_EXCEPTION_H

#include <cstdlib>
#include "core/Exception.h"

namespace bob { namespace ip {

  class Exception: public bob::core::Exception {

    public:
      Exception() throw();
      virtual ~Exception() throw();
      virtual const char* what() const throw();

  };

  class ParamOutOfBoundaryError: public Exception {
    public:
      ParamOutOfBoundaryError(const std::string& paramname, const bool larger, 
        const int value, const int limit) throw();
      virtual ~ParamOutOfBoundaryError() throw();
      virtual const char* what() const throw();

    private:
      std::string m_paramname;
      bool m_larger;
      int m_value;
      int m_limit;
      mutable std::string m_message;
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
