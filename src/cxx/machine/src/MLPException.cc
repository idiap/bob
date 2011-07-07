/**
 * @author Andre Anjos <andre.anjos@idiap.ch> 
 * @date Thu  7 Jul 13:42:06 2011
 *
 * @brief Implementation of MLP specific exceptions
 */

#include <boost/format.hpp>
#include "machine/MLPException.h"

namespace mach = Torch::machine;

mach::InvalidShape::InvalidShape() throw() {
}

mach::InvalidShape::~InvalidShape() throw() {
}

const char* mach::InvalidShape::what() const throw() {
 static const char* what_string = "Trying to setup an MLP with a shape containing only 1 entry. You have to provide at least 2.";
 return what_string;
}

mach::NumberOfLayersMismatch::NumberOfLayersMismatch(size_t expected,
    size_t got) throw():
  m_expected(expected),
  m_got(got)
{
}

mach::NumberOfLayersMismatch::~NumberOfLayersMismatch() throw() {
}

const char* mach::NumberOfLayersMismatch::what() const throw() {
  try {
    boost::format message("Mismatch in the number of layers. I expected %d, but you are trying to set %d.");
    message % m_expected % m_got;
    m_message = message.str();
    return m_message.c_str();
  } catch (...) {
    static const char* emergency = "mach::NumberOfLayersMismatch: cannot format, exception raised";
    return emergency;
  }
}

mach::WeightShapeMismatch::WeightShapeMismatch(size_t layer, 
    const blitz::TinyVector<int,2>& expected,
    const blitz::TinyVector<int,2>& given) throw():
  m_layer(layer),
  m_expected(expected),
  m_given(given)
{
}

mach::WeightShapeMismatch::~WeightShapeMismatch() throw() {
}

const char* mach::WeightShapeMismatch::what() const throw() {
  try {
    boost::format message("Mismatch in the expected shape for the weights in layer %d. I got (%d,%d) but I was expecting (%d,%d).");
    message % m_layer;
    message % m_expected[0] % m_expected[1];
    message % m_given[0] % m_given[1];
    m_message = message.str();
    return m_message.c_str();
  } catch (...) {
    static const char* emergency = "mach::WeightShapeMismatch: cannot format, exception raised";
    return emergency;
  }
}

mach::BiasShapeMismatch::BiasShapeMismatch(size_t layer, 
    size_t expected, size_t given) throw():
  m_layer(layer),
  m_expected(expected),
  m_given(given)
{
}

mach::BiasShapeMismatch::~BiasShapeMismatch() throw() {
}

const char* mach::BiasShapeMismatch::what() const throw() {
  try {
    boost::format message("Mismatch in the expected shape for the bias in layer %d. I got shape (%d) but I was expecting (%d).");
    message % m_layer;
    message % m_expected % m_given;
    m_message = message.str();
    return m_message.c_str();
  } catch (...) {
    static const char* emergency = "mach::BiasShapeMismatch: cannot format, exception raised";
    return emergency;
  }
}
