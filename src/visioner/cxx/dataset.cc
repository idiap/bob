/**
 * @file visioner/cxx/dataset.cc
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

#include "bob/visioner/model/dataset.h"

namespace bob { namespace visioner {

  // Constructor
  DataSet::DataSet(uint64_t n_outputs, uint64_t n_samples, uint64_t n_features, uint64_t n_fvalues)
    :	m_n_fvalues(n_fvalues)
  {
    resize(n_outputs, n_samples, n_features, n_fvalues);
  }

  // Resize
  void DataSet::resize(uint64_t n_outputs, uint64_t n_samples, uint64_t n_features, uint64_t n_fvalues)
  {
    m_n_fvalues = n_fvalues;
    m_targets.resize(n_samples, n_outputs);
    m_values.resize(n_features, n_samples);
    m_costs.resize(n_samples);
  }

}}
