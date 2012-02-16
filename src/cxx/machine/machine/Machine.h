/**
 * @file cxx/machine/machine/Machine.h
 * @date Tue Jan 18 17:07:26 2011 +0100
 * @author André Anjos <andre.anjos@idiap.ch>
 *
 * Copyright (C) 2011-2012 Idiap Reasearch Institute, Martigny, Switzerland
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

#ifndef BOB_MACHINE_MACHINE_H
#define BOB_MACHINE_MACHINE_H

#include <cstring>

namespace bob { namespace machine {

/**
 * Root class for all machines
 */
template<class T_input, class T_output>
class Machine
{
  public:
    virtual ~Machine() {}

    /**
     * Execute the machine
     *
     * @param input input data used by the machine
     * @param output value computed by the machine
     * @warning Inputs are checked
     */
    virtual void forward(const T_input& input, T_output& output) const = 0;

    /**
     * Execute the machine
     *
     * @param input input data used by the machine
     * @param output value computed by the machine
     * @warning Inputs are NOT checked
     */
    virtual void forward_(const T_input& input, T_output& output) const = 0;
};

}}
#endif 
