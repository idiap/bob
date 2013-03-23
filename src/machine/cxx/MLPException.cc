/**
 * @file machine/cxx/MLPException.cc
 * @date Thu Jul 7 16:49:35 2011 +0200
 * @author Andre Anjos <andre.anjos@idiap.ch>
 *
 * @brief Implementation of MLP specific exceptions
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

#include <boost/format.hpp>
#include <bob/machine/MLPException.h>

bob::machine::InvalidShape::InvalidShape() throw() {
}

bob::machine::InvalidShape::~InvalidShape() throw() {
}

const char* bob::machine::InvalidShape::what() const throw() {
 static const char* what_string = "Trying to setup an MLP with a shape containing only 1 entry. You have to provide at least 2.";
 return what_string;
}

bob::machine::NumberOfLayersMismatch::NumberOfLayersMismatch(size_t expected,
    size_t got) throw():
  m_expected(expected),
  m_got(got)
{
}

bob::machine::NumberOfLayersMismatch::~NumberOfLayersMismatch() throw() {
}

const char* bob::machine::NumberOfLayersMismatch::what() const throw() {
  try {
    boost::format message("Mismatch in the number of layers. I expected %d, but you are trying to set %d.");
    message % m_expected % m_got;
    m_message = message.str();
    return m_message.c_str();
  } catch (...) {
    static const char* emergency = "bob::machine::NumberOfLayersMismatch: cannot format, exception raised";
    return emergency;
  }
}

bob::machine::WeightShapeMismatch::WeightShapeMismatch(size_t layer, 
    const blitz::TinyVector<int,2>& expected,
    const blitz::TinyVector<int,2>& given) throw():
  m_layer(layer),
  m_expected(expected),
  m_given(given)
{
}

bob::machine::WeightShapeMismatch::~WeightShapeMismatch() throw() {
}

const char* bob::machine::WeightShapeMismatch::what() const throw() {
  try {
    boost::format message("Mismatch in the expected shape for the weights in layer %d. I got (%d,%d) but I was expecting (%d,%d).");
    message % m_layer;
    message % m_expected[0] % m_expected[1];
    message % m_given[0] % m_given[1];
    m_message = message.str();
    return m_message.c_str();
  } catch (...) {
    static const char* emergency = "bob::machine::WeightShapeMismatch: cannot format, exception raised";
    return emergency;
  }
}

bob::machine::BiasShapeMismatch::BiasShapeMismatch(size_t layer, 
    size_t expected, size_t given) throw():
  m_layer(layer),
  m_expected(expected),
  m_given(given)
{
}

bob::machine::BiasShapeMismatch::~BiasShapeMismatch() throw() {
}

const char* bob::machine::BiasShapeMismatch::what() const throw() {
  try {
    boost::format message("Mismatch in the expected shape for the bias in layer %d. I got shape (%d) but I was expecting (%d).");
    message % m_layer;
    message % m_expected % m_given;
    m_message = message.str();
    return m_message.c_str();
  } catch (...) {
    static const char* emergency = "bob::machine::BiasShapeMismatch: cannot format, exception raised";
    return emergency;
  }
}

bob::machine::UnsupportedActivation::UnsupportedActivation(bob::machine::Activation act) 
  throw():
  m_act(act)
{
}

bob::machine::UnsupportedActivation::~UnsupportedActivation() throw() {
}

const char* bob::machine::UnsupportedActivation::what() const throw() {
  try {
    boost::format message("Object does not support use of activation function %d.");
    message % (unsigned)m_act;
    m_message = message.str();
    return m_message.c_str();
  } catch (...) {
    static const char* emergency = "bob::machine::UnsupportedActivation: cannot format, exception raised";
    return emergency;
  }
}
