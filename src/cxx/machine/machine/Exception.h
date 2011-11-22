/**
 * @file cxx/machine/machine/Exception.h
 * @date Fri May 13 20:22:43 2011 +0200
 * @author Laurent El Shafey <Laurent.El-Shafey@idiap.ch>
 *
  * @brief Exceptions used throughout the machine subsystem of Torch
 *
 * Copyright (C) 2011 Idiap Reasearch Institute, Martigny, Switzerland
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

#ifndef TORCH5SPRO_MACHINE_EXCEPTION_H 
#define TORCH5SPRO_MACHINE_EXCEPTION_H

#include <cstdlib>
#include "core/Exception.h"

namespace Torch { namespace machine {

  class Exception: public Torch::core::Exception {

    public:
      Exception() throw();
      virtual ~Exception() throw();
      virtual const char* what() const throw();

  };

  class NInputsMismatch: public Exception {
    public:
      NInputsMismatch(const int n_inputs1, const int n_inputs2) throw();
      virtual ~NInputsMismatch() throw();
      virtual const char* what() const throw();

    private:
      int m_n_inputs1;
      int m_n_inputs2;
      mutable std::string m_message;
  };

  class NOutputsMismatch: public Exception {
    public:
      NOutputsMismatch(const int n_outputs1, const int n_outputs2) throw();
      virtual ~NOutputsMismatch() throw();
      virtual const char* what() const throw();

    private:
      int m_n_outputs1;
      int m_n_outputs2;
      mutable std::string m_message;
  };

}}

#endif /* TORCH5SPRO_MACHINE_EXCEPTION_H */
