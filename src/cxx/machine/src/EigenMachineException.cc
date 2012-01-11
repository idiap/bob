/**
 * @file cxx/machine/src/EigenMachineException.cc
 * @date Fri May 13 20:22:43 2011 +0200
 * @author Laurent El Shafey <Laurent.El-Shafey@idiap.ch>
 *
  * @brief Implements the exceptions for the EigenMachine
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

#include "machine/EigenMachineException.h"
#include <boost/format.hpp>

namespace machine = bob::machine;

machine::EigenMachineNOutputsTooLarge::EigenMachineNOutputsTooLarge(const int n_outputs, const int n_outputs_max) throw(): 
  m_n_outputs(n_outputs), m_n_outputs_max(n_outputs_max) 
{
}

machine::EigenMachineNOutputsTooLarge::~EigenMachineNOutputsTooLarge() throw() {
}

const char* machine::EigenMachineNOutputsTooLarge::what() const throw() {
  try {
    boost::format message("Trying to set a too large number of outputs '%d', as only '%d' eigenvalues/eigenvectors have been set in the machine.");
    message % m_n_outputs;
    message % m_n_outputs_max;
    m_message = message.str();
    return m_message.c_str();
  } catch (...) {
    static const char* emergency = "machine::EigenMachineNOutputsTooLarge: cannot format, exception raised";
    return emergency;
  }
}

