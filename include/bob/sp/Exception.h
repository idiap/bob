/**
 * @file bob/sp/Exception.h
 * @date Tue Feb 5 20:41:55 2011 +0100
 * @author Laurent El Shafey <Laurent.El-Shafey@idiap.ch>
 *
 * @brief Exceptions used throughout the SP subsystem of bob
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

#ifndef BOB_SP_EXCEPTION_H 
#define BOB_SP_EXCEPTION_H

#include "bob/core/Exception.h"

namespace bob { namespace sp {

  class Exception: public bob::core::Exception {

    public:
      Exception() throw();
      virtual ~Exception() throw();
      virtual const char* what() const throw();

  };

  class ExtrapolationDstTooSmall: public Exception {
    public:
      ExtrapolationDstTooSmall() throw();
      virtual ~ExtrapolationDstTooSmall() throw();
      virtual const char* what() const throw();

    private:
      mutable std::string m_message;
  };

  class ConvolutionKernelTooLarge: public Exception {
    public:
      ConvolutionKernelTooLarge(int dim, int size_array, int size_kernel) throw();
      virtual ~ConvolutionKernelTooLarge() throw();
      virtual const char* what() const throw();

    private:
      mutable std::string m_message;
      int m_dim;
      int m_size_array;
      int m_size_kernel;
  };

  class SeparableConvolutionInvalidDim: public Exception {
    public:
      SeparableConvolutionInvalidDim(int dim, int max_dim) throw();
      virtual ~SeparableConvolutionInvalidDim() throw();
      virtual const char* what() const throw();

    private:
      mutable std::string m_message;
      int m_dim;
      int m_max_dim;
  };


}}

#endif /* BOB_SP_EXCEPTION_H */
