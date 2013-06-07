/**
 * @author Andre Anjos <andre.anjos@idiap.ch>
 * @date Fri 17 May 2013 12:29:19 CEST
 *
 * @brief Implementation of activation functions
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

#include <cmath>
#include <boost/make_shared.hpp>
#include "bob/machine/Activation.h"
#include "bob/machine/ActivationRegistry.h"

namespace bob { namespace machine {

  double IdentityActivation::f (double z) const { return z; }

  double IdentityActivation::f_prime (double) const { return 1.; }
  
  double IdentityActivation::f_prime_from_f (double) const { return 1.; }

  void IdentityActivation::save(bob::io::HDF5File& f) const {
    f.set("id", unique_identifier());
  }

  void IdentityActivation::load(bob::io::HDF5File&) {
  }

  std::string IdentityActivation::unique_identifier() const {
    return "bob.machine.Activation.Identity";
  }

  std::string IdentityActivation::str() const {
    return "f(z) = z";
  }

  LinearActivation::LinearActivation(double C): m_C(C) {}

  LinearActivation::~LinearActivation() {}

  double LinearActivation::f (double z) const { return m_C * z; }

  double LinearActivation::f_prime (double z) const { return m_C; }
  
  double LinearActivation::f_prime_from_f (double a) const { return m_C; }

  double LinearActivation::C() const { return m_C; }

  void LinearActivation::save(bob::io::HDF5File& f) const {
    f.set("id", unique_identifier());
    f.set("C", m_C);
  }

  void LinearActivation::load(bob::io::HDF5File& f) {
    m_C = f.read<double>("C");
  }

  std::string LinearActivation::unique_identifier() const {
    return "bob.machine.Activation.Linear";
  }

  std::string LinearActivation::str() const {
    boost::format m("f(z) = %.5e * z");
    m % m_C;
    return m.str();
  }

  double HyperbolicTangentActivation::f (double z) const { return std::tanh(z); }

  double HyperbolicTangentActivation::f_prime (double z) const { return f_prime_from_f(f(z)); }

  double HyperbolicTangentActivation::f_prime_from_f (double a) const { return (1. - (a*a)); }

  void HyperbolicTangentActivation::save(bob::io::HDF5File& f) const {
    f.set("id", unique_identifier());
  }

  void HyperbolicTangentActivation::load(bob::io::HDF5File&) {
  }

  std::string HyperbolicTangentActivation::unique_identifier() const {
    return "bob.machine.Activation.HyperbolicTangent";
  }

  std::string HyperbolicTangentActivation::str() const {
    return "f(z) = tanh(z)";
  }

  MultipliedHyperbolicTangentActivation::MultipliedHyperbolicTangentActivation(double C, double M) : m_C(C), m_M(M) {}

  MultipliedHyperbolicTangentActivation::~MultipliedHyperbolicTangentActivation() {}

  double MultipliedHyperbolicTangentActivation::f (double z) const { return m_C * std::tanh(m_M * z); }

  double MultipliedHyperbolicTangentActivation::f_prime (double z) const
  { return f_prime_from_f(f(z)); }

  double MultipliedHyperbolicTangentActivation::f_prime_from_f (double a) const
  { return m_C * m_M * (1. - std::pow(a/m_C,2)); }

  double MultipliedHyperbolicTangentActivation::C() const { return m_C; }

  double MultipliedHyperbolicTangentActivation::M() const { return m_M; }

  void MultipliedHyperbolicTangentActivation::save(bob::io::HDF5File& f) const {
    f.set("id", unique_identifier());
    f.set("C", m_C);
    f.set("M", m_C);
  }

  void MultipliedHyperbolicTangentActivation::load(bob::io::HDF5File& f) {
    m_C = f.read<double>("C");
    m_M = f.read<double>("M");
  }

  std::string MultipliedHyperbolicTangentActivation::unique_identifier() const {
    return "bob.machine.Activation.MultipliedHyperbolicTangent";
  }

  std::string MultipliedHyperbolicTangentActivation::str() const {
    boost::format m("f(z) = %.5e * tanh(%.5e * z)");
    m % m_C % m_M;
    return m.str();
  }

  double LogisticActivation::f (double z) const 
  { return 1. / ( 1. + std::exp(-z) ); }

  double LogisticActivation::f_prime (double z) const { return f_prime_from_f(f(z)); }

  double LogisticActivation::f_prime_from_f (double a) const { return a * (1. - a); }

  void LogisticActivation::save(bob::io::HDF5File& f) const {
    f.set("id", unique_identifier());
  }

  void LogisticActivation::load(bob::io::HDF5File&) {
  }

  std::string LogisticActivation::unique_identifier() const {
    return "bob.machine.Activation.Logistic";
  }

  std::string LogisticActivation::str() const {
    return "f(z) = 1./(1. + e^-z)";
  }

  boost::shared_ptr<Activation> load_activation(bob::io::HDF5File& f) {
    auto make = ActivationRegistry::instance()->find(f.read<std::string>("id"));
    return make(f);
  }

  boost::shared_ptr<Activation> make_deprecated_activation(uint32_t e) {
    switch(e) {
      case 0:
        return boost::make_shared<IdentityActivation>();
        break;
      case 1:
        return boost::make_shared<HyperbolicTangentActivation>();
        break;
      case 2:
        return boost::make_shared<LogisticActivation>();
        break;
      default:
        throw std::runtime_error("unsupported (deprecated) activation read from HDF5 file - not any of 0 (linear), 1 (tanh) or 2 (logistic)");
    }
  }

}}

/**
 * A generalized registration mechanism for all classes above
 */
template <typename T> struct register_activation {
  
  static boost::shared_ptr<bob::machine::Activation> factory 
    (bob::io::HDF5File& f) {
      auto retval = boost::make_shared<T>();
      retval->load(f);
      return retval;
    }

  register_activation() {
    T obj;
    bob::machine::ActivationRegistry::instance()->registerActivation
      (obj.unique_identifier(), register_activation<T>::factory);
  }

};

// register all extensions
static register_activation<bob::machine::IdentityActivation> _identity_act_reg;
static register_activation<bob::machine::LinearActivation> _linear_act_reg;
static register_activation<bob::machine::HyperbolicTangentActivation> _tanh_act_reg;
static register_activation<bob::machine::MultipliedHyperbolicTangentActivation> _multanh_act_reg;
static register_activation<bob::machine::LogisticActivation> _logistic_act_reg;
