/**
 * @file cxx/trainer/trainer/Exception.h
 * @date Wed May 18 16:14:44 2011 +0200
 * @author Francois Moulin <Francois.Moulin@idiap.ch>
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
#ifndef TORCH5SPRO_TRAINER_EXCEPTION_H
#define TORCH5SPRO_TRAINER_EXCEPTION_H

#include "core/Exception.h"

namespace Torch { namespace trainer {

  class Exception: public Torch::core::Exception {

    public:
      Exception() throw();
      virtual ~Exception() throw();
      virtual const char* what() const throw();

  };

  /**
   * Raised when some computations need a prior GMM and no one is set
   */
  class NoPriorGMM: public Exception {
    public:
      NoPriorGMM() throw();
      virtual ~NoPriorGMM() throw();
      virtual const char* what() const throw();
  };

  /**
   * Raised when the number of classes is insufficient.
   */
  class WrongNumberOfClasses: public Exception {
    public:
      WrongNumberOfClasses(size_t got) throw();
      virtual ~WrongNumberOfClasses() throw();
      virtual const char* what() const throw();

    private:
      size_t m_got;
      mutable std::string m_message;
  };

  /**
   * Raised when the number of features is different between classes
   */
  class WrongNumberOfFeatures: public Exception {
    public:
      WrongNumberOfFeatures(size_t got, size_t expected,
          size_t classid) throw();
      virtual ~WrongNumberOfFeatures() throw();
      virtual const char* what() const throw();

    private:
      size_t m_got;
      size_t m_expected;
      size_t m_classid;
      mutable std::string m_message;
  };

  /**
   * Raised when the given machine is incompatible with the current settings
   * for a given trainer.
   */
  class IncompatibleMachine: public Exception {
    public:
      IncompatibleMachine() throw();
      virtual ~IncompatibleMachine() throw();
      virtual const char* what() const throw();
  };

}}

#endif /* TORCH5SPRO_TRAINER_EXCEPTION_H */
