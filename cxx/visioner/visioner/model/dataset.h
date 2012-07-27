/**
 * @file visioner/visioner/model/dataset.h
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

#ifndef BOB_VISIONER_DATASET_H
#define BOB_VISIONER_DATASET_H

#include "visioner/model/ml.h"

namespace bob { namespace visioner {

  ////////////////////////////////////////////////////////////////////////////////
  // Dataset where the feature values are stored in memory.
  // Storage:
  //	- targets:		#outputs x #samples
  //	- feature values:	#features x #samples
  ////////////////////////////////////////////////////////////////////////////////

  class DataSet
  {
    public:

      // Constructor
      DataSet(index_t n_outputs = 0, index_t n_samples = 0,
          index_t n_features = 0, index_t n_fvalues = 0);

      // Resize
      void resize(index_t n_outputs, index_t n_samples,
          index_t n_features, index_t n_fvalues);

      // Access functions
      bool empty() const { return m_targets.empty(); }
      index_t n_outputs() const { return m_targets.cols(); }
      index_t n_samples() const { return m_targets.rows(); }
      index_t	n_features() const { return m_values.rows(); }
      index_t n_fvalues() const { return m_n_fvalues; }

      scalar_t target(index_t s, index_t o) const { return m_targets(s, o); }
      scalar_t& target(index_t s, index_t o) { return m_targets(s, o); }
      const scalar_mat_t& targets() const { return m_targets; }

      discrete_t value(index_t f, index_t s) const { return m_values(f, s); }
      discrete_t& value(index_t f, index_t s) { return m_values(f, s); }
      const discrete_mat_t& values() const { return m_values; }

      scalar_t cost(index_t s) const { return m_costs[s]; }
      scalar_t& cost(index_t s) { return m_costs[s]; }
      const scalars_t& costs() const { return m_costs; }

    private:

      // Attributes
      index_t		m_n_fvalues;
      scalar_mat_t	m_targets;
      discrete_mat_t	m_values;
      scalars_t       m_costs;
  };

}}

#endif // BOB_VISIONER_DATASET_H
