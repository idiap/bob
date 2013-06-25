/**
 * @file bob/machine/roll.h
 * @date Tue Jun 25 18:48:20 CEST 2013
 * @author Laurent El Shafey <Laurent.El-Shafey@idiap.ch>
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

#ifndef BOB_MACHINE_ROLL_H 
#define BOB_MACHINE_ROLL_H 

#include <vector>
#include <blitz/array.h>
#include <bob/machine/MLP.h>

namespace bob { namespace machine {
/**
 * @ingroup MACHINE
 * @{
 */

namespace detail {
  /**
   * @brief Returns the number of parameters (weights and biases) in an
   * MLP.
   */
  int getNbParameters(const bob::machine::MLP& machine);
  /**
   * @brief Returns the number of parameters (weights and biases).
   */
  int getNbParameters(const std::vector<blitz::Array<double,2> >& weights,
    const std::vector<blitz::Array<double,1> >& biases);
}

/**
 * @brief Puts the parameters (weights and biases) of the machine in a 
 * large single 1D vector
 */
void unroll(const bob::machine::MLP& machine, blitz::Array<double,1>& vec);
/**
 * @brief Puts the parameters (weights and biases) in a large single 1D vector
 */
void unroll(const std::vector<blitz::Array<double,2> >& weights, 
  const std::vector<blitz::Array<double,1> >& biases,
  blitz::Array<double,1>& vec);

/**
 * @brief Sets the parameters (weights and biases) of the machine from a
 * large single 1D vector
 */
void roll(bob::machine::MLP& machine, const blitz::Array<double,1>& vec);

/**
 * @brief Sets the parameters (weights and biases) from a
 * large single 1D vector
 */
void roll(std::vector<blitz::Array<double,2> >& weights,
  std::vector<blitz::Array<double,1> >& biases, 
  const blitz::Array<double,1>& vec);

/**
 * @}
 */
}}

#endif /* BOB_MACHINE_ROLL_H */
