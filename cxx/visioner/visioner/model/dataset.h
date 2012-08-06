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
      DataSet(uint64_t n_outputs = 0, uint64_t n_samples = 0,
          uint64_t n_features = 0, uint64_t n_fvalues = 0);

      // Resize
      void resize(uint64_t n_outputs, uint64_t n_samples,
          uint64_t n_features, uint64_t n_fvalues);

      // Access functions
      bool empty() const { return m_targets.empty(); }
      uint64_t n_outputs() const { return m_targets.cols(); }
      uint64_t n_samples() const { return m_targets.rows(); }
      uint64_t	n_features() const { return m_values.rows(); }
      uint64_t n_fvalues() const { return m_n_fvalues; }

      double target(uint64_t s, uint64_t o) const { return m_targets(s, o); }
      double& target(uint64_t s, uint64_t o) { return m_targets(s, o); }
      const Matrix<double>& targets() const { return m_targets; }

      uint16_t value(uint64_t f, uint64_t s) const { return m_values(f, s); }
      uint16_t& value(uint64_t f, uint64_t s) { return m_values(f, s); }
      const Matrix<uint16_t>& values() const { return m_values; }

      double cost(uint64_t s) const { return m_costs[s]; }
      double& cost(uint64_t s) { return m_costs[s]; }
      const std::vector<double>& costs() const { return m_costs; }

    private:

      // Attributes
      uint64_t		m_n_fvalues;
      Matrix<double>	m_targets;
      Matrix<uint16_t>	m_values;
      std::vector<double>       m_costs;
  };

}}

#endif // BOB_VISIONER_DATASET_H
