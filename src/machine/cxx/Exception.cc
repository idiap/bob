/**
 * @file machine/cxx/Exception.cc
 * @date Fri May 13 20:22:43 2011 +0200
 * @author Laurent El Shafey <Laurent.El-Shafey@idiap.ch>
 *
  * @brief Implements a generic exception for the machine subsystem of bob
 * @brief
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

#include <bob/machine/Exception.h>
#include <boost/format.hpp>

bob::machine::Exception::Exception() throw() {
}

bob::machine::Exception::~Exception() throw() {
}

const char* bob::machine::Exception::what() const throw() {
 static const char* what_string = "Generic bob::machine::Exception: user specialization has not overwritten what() or is throwing an object of this class (in which case, please fix it!)";
 return what_string;
}

bob::machine::NInputsMismatch::NInputsMismatch(const int n1, const int n2) throw(): m_n_inputs1(n1), m_n_inputs2(n2) {
}

bob::machine::NInputsMismatch::~NInputsMismatch() throw() {
}

const char* bob::machine::NInputsMismatch::what() const throw() {
  try {
    boost::format message("Mismatch in the number of inputs: '%d' vs '%d'.");
    message % m_n_inputs1;
    message % m_n_inputs2;
    m_message = message.str();
    return m_message.c_str();
  } catch (...) {
    static const char* emergency = "bob::machine::NInputsMismatch: cannot format, exception raised";
    return emergency;
  }
}

bob::machine::NOutputsMismatch::NOutputsMismatch(const int n1, const int n2) throw(): m_n_outputs1(n1), m_n_outputs2(n2) {
}

bob::machine::NOutputsMismatch::~NOutputsMismatch() throw() {
}

const char* bob::machine::NOutputsMismatch::what() const throw() {
  try {
    boost::format message("Mismatch in the number of outputs: '%d' vs '%d'.");
    message % m_n_outputs1;
    message % m_n_outputs2;
    m_message = message.str();
    return m_message.c_str();
  } catch (...) {
    static const char* emergency = "bob::machine::NOutputsMismatch: cannot format, exception raised";
    return emergency;
  }
}
