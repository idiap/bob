/**
 * @file visioner/visioner/model/tagger.h
 * @date Fri 27 Jul 13:58:57 2012 CEST
 * @author Andre Anjos <andre.anjos@idiap.ch>
 *
 * @brief This file was part of Visioner and originally authored by "Cosmin
 * Atanasoaei <cosmin.atanasoaei@idiap.ch>". It was only modified to conform to
 * Bob coding standards and structure.
 *
 * Copyright (C) 2011-2012 Idiap Research Institute, Martigny, Switzerland
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

#ifndef BOB_VISIONER_TAGGER_H
#define BOB_VISIONER_TAGGER_H

#include "bob/visioner/model/ipyramid.h"
#include "bob/visioner/model/param.h"

namespace bob { namespace visioner {

  /**
   * Sub-window labelling for either classification or regression.
   */
  class Tagger : public Parametrizable {

    public:

      // Constructor
      Tagger(const param_t& param = param_t())
        :	Parametrizable(param)
      {
      }

      // Destructor
      virtual ~Tagger() {}

      // Clone the object
      virtual boost::shared_ptr<Tagger> clone() const = 0;

      // Reset to new parameters
      virtual void reset(const param_t& param) { m_param = param; }

      // Number of outputs 
      virtual uint64_t n_outputs() const = 0;

      // Number of types
      virtual uint64_t n_types() const = 0;

      // Label a sub-window
      virtual bool check(const ipscale_t& ipscale, int x, int y, 
          std::vector<double>& targets, uint64_t& type) const = 0;
  };

}}

#endif // BOB_VISIONER_TAGGER_H
