/**
 * @date Thu Jul 7 16:49:35 2011 +0200
 * @author Andre Anjos <andre.anjos@idiap.ch>
 *
 * @brief Activation functions for linear and MLP machines.
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

#ifndef BOB_MACHINE_ACTIVATION_H 
#define BOB_MACHINE_ACTIVATION_H

#include <string>
#include <boost/shared_ptr.hpp>
#include "bob/io/HDF5File.h"

namespace bob { namespace machine {
  /**
   * @ingroup MACHINE
   * @{
   */

  /**
   * Base class for activation functions. All activation functions must derive
   * from this one.
   */
  class Activation {

    public: // api

      /**
       * Computes activated value, given an input.
       */
      virtual double f (double z) const =0;

      /**
       * Computes the derivative of the activated value, given the activation
       * value used to compute the activated value originally.
       */
      virtual double f_prime (double z) const =0;

      /**
       * Saves itself to an HDF5File
       */
      virtual void save(bob::io::HDF5File& f) const =0;

      /**
       * Loads itself from an HDF5File
       */
      virtual void load(bob::io::HDF5File& f) =0;

      /**
       * Returns a unique identifier, used by this class in connection to the
       * Activation registry.
       */
      virtual std::string unique_identifier() const =0;

      /**
       * Returns a stringified representation for this Activation function
       */
      virtual std::string str() const =0;

  };

  /**
   * Generic interface for Activation object factories
   */
  typedef boost::shared_ptr<Activation> (*activation_factory_t)
    (bob::io::HDF5File& f);

  /**
   * Loads an activation function from file using the new API
   */
  boost::shared_ptr<Activation> load_activation(bob::io::HDF5File& f);

  /**
   * Loads an activation function using the old API
   *
   * @param e The old enumeration value for activation functions:
   *        (0) - linear; (1) - tanh; (2) - logistic
   */
  boost::shared_ptr<Activation> make_deprecated_activation(uint32_t e);

  /**
   * Implements the activation function f(z) = z
   */
  class IdentityActivation: public Activation {

    public: // api

      virtual double f (double z) const;
      virtual double f_prime (double) const;
      virtual void save(bob::io::HDF5File&) const;
      virtual void load(bob::io::HDF5File&);
      virtual std::string unique_identifier() const;
      virtual std::string str() const;

  };

  /**
   * Implements the activation function f(z) = C*z
   */
  class LinearActivation: public Activation {

    public: // api

      LinearActivation(double C=1.);
      virtual ~LinearActivation();
      virtual double f (double z) const;
      virtual double f_prime (double z) const;
      double C() const;
      virtual void save(bob::io::HDF5File& f) const;
      virtual void load(bob::io::HDF5File&);
      virtual std::string unique_identifier() const;
      virtual std::string str() const;

    private: // representation

      double m_C; ///< multiplication factor

  };

  /**
   * Implements the activation function f(z) = std::tanh(z)
   */
  class HyperbolicTangentActivation: public Activation {

    public: // api

      virtual double f (double z) const;
      virtual double f_prime (double z) const;
      virtual void save(bob::io::HDF5File& f) const;
      virtual void load(bob::io::HDF5File&);
      virtual std::string unique_identifier() const;
      virtual std::string str() const;

  };

  /**
   * Implements the activation function f(z) = C*std::tanh(M*z)
   */
  class MultipliedHyperbolicTangentActivation: public Activation {

    public: // api

      MultipliedHyperbolicTangentActivation(double C=1., double M=1.);
      virtual ~MultipliedHyperbolicTangentActivation();
      virtual double f (double z) const;
      virtual double f_prime (double z) const; 
      double C() const;
      double M() const;
      virtual void save(bob::io::HDF5File& f) const;
      virtual void load(bob::io::HDF5File& f);
      virtual std::string unique_identifier() const;
      virtual std::string str() const;

    private: // representation

      double m_C; ///< multiplication factor
      double m_M; ///< internal multiplication factor

  };

  /**
   * Implements the activation function f(z) = 1. / ( 1. + e^(-z) )
   */
  class LogisticActivation: public Activation {

    public: // api

      virtual double f (double z) const;
      virtual double f_prime (double z) const;
      virtual void save(bob::io::HDF5File& f) const;
      virtual void load(bob::io::HDF5File&);
      virtual std::string unique_identifier() const;
      virtual std::string str() const;

  };

  /**
   * @}
   */
}}
      
#endif /* BOB_MACHINE_ACTIVATION_H */
