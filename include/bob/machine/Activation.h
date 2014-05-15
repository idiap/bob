/**
 * @date Thu Jul 7 16:49:35 2011 +0200
 * @author Andre Anjos <andre.anjos@idiap.ch>
 *
 * @brief Activation functions for linear and MLP machines.
 *
 * Copyright (C) Idiap Research Institute, Martigny, Switzerland
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
       * Computes the derivative of the current activation - i.e., the same
       * input as for f().
       */
      virtual double f_prime (double z) const =0;

      /**
       * Computes the derivative of the activated value, given the activated
       * value - that is, the output of Activation::f() above.
       */
      virtual double f_prime_from_f (double a) const =0;

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

      virtual ~IdentityActivation();
      virtual double f (double z) const;
      virtual double f_prime (double z) const;
      virtual double f_prime_from_f (double a) const;
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
      virtual double f_prime_from_f (double a) const;
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

      virtual ~HyperbolicTangentActivation();
      virtual double f (double z) const;
      virtual double f_prime (double z) const;
      virtual double f_prime_from_f (double a) const;
      virtual void save(bob::io::HDF5File& f) const;
      virtual void load(bob::io::HDF5File&);
      virtual std::string unique_identifier() const;
      virtual std::string str() const;

  };

  /**
   * Implements the activation function f(z) = C*tanh(M*z)
   */
  class MultipliedHyperbolicTangentActivation: public Activation {

    public: // api

      MultipliedHyperbolicTangentActivation(double C=1., double M=1.);
      virtual ~MultipliedHyperbolicTangentActivation();
      virtual double f (double z) const;
      virtual double f_prime (double z) const;
      virtual double f_prime_from_f (double a) const;
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

      virtual ~LogisticActivation();
      virtual double f (double z) const;
      virtual double f_prime (double z) const;
      virtual double f_prime_from_f (double a) const;
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
