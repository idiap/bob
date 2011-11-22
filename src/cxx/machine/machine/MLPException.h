/**
 * @file cxx/machine/machine/MLPException.h
 * @date Thu Jul 7 16:49:35 2011 +0200
 * @author Andre Anjos <andre.anjos@idiap.ch>
 *
 * @brief Specific exceptions to MLPs
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

#ifndef TORCH_MACHINE_MLPEXCEPTION_H 
#define TORCH_MACHINE_MLPEXCEPTION_H

#include <blitz/array.h>
#include "machine/Exception.h"
#include "machine/Activation.h"

namespace Torch { namespace machine {

  /**
   * Exception raised when the resizing shape has less than 2 components
   */
  class InvalidShape: public Exception {
    public:
      InvalidShape() throw();
      virtual ~InvalidShape() throw();
      virtual const char* what() const throw();
  };

  /**
   * Exception raised when there is a mismatch between the number of layers
   */
  class NumberOfLayersMismatch: public Exception {
    public:
      NumberOfLayersMismatch(size_t expected, size_t got) throw(); 
      virtual ~NumberOfLayersMismatch() throw();
      virtual const char* what() const throw();
      
    private:
      size_t m_expected;
      size_t m_got;
      mutable std::string m_message;
  };

  /**
   * Exception raised when there is a mismatch between the shapes of weights to
   * be set and the current MLP size.
   */
  class WeightShapeMismatch: public Exception {
    public:
      WeightShapeMismatch(size_t layer, 
          const blitz::TinyVector<int,2>& expected,
          const blitz::TinyVector<int,2>& given) throw();

      virtual ~WeightShapeMismatch() throw();
      virtual const char* what() const throw();
      
    private:
      size_t m_layer;
      blitz::TinyVector<int,2> m_expected;
      blitz::TinyVector<int,2> m_given;
      mutable std::string m_message;
  };

  /**
   * Exception raised when there is a mismatch between the shapes of biases to
   * be set and the current MLP size.
   */
  class BiasShapeMismatch: public Exception {
    public:
      BiasShapeMismatch(size_t layer, size_t expected, size_t given) throw();
      virtual ~BiasShapeMismatch() throw();
      virtual const char* what() const throw();
      
    private:
      size_t m_layer;
      size_t m_expected;
      size_t m_given;
      mutable std::string m_message;
  };

  /**
   * Exception raised when machine (or trainers) do not support a certain
   * activation function.
   */
  class UnsupportedActivation: public Exception {
    public:
      UnsupportedActivation(Torch::machine::Activation act) throw();
      virtual ~UnsupportedActivation() throw();
      virtual const char* what() const throw();
      
    private:
      Torch::machine::Activation m_act;
      mutable std::string m_message;
  };

}}

#endif /* TORCH_MACHINE_MLPEXCEPTION_H */

