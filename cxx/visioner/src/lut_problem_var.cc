/**
 * @file visioner/src/lut_problem_var.cc
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

#include <numeric>
#include <omp.h>

#include "visioner/model/trainers/lutproblems/lut_problem_var.h"

namespace bob { namespace visioner {

  // Constructor
  LUTProblemVAR::LUTProblemVAR(const DataSet& data, const param_t& param, double lambda, size_t threads)
    :       LUTProblemEPT(data, param, threads), 
    m_lambda(lambda)
  {                
  }

  // Update loss values and derivatives (for some particular scores)
  void LUTProblemVAR::update_loss_deriv(const Matrix<double>& scores)
  {
    // Compute the expectation loss values and derivatives
    LUTProblemEPT::update_loss_deriv(scores);

    const double scale1 = m_lambda * n_samples();
    const double scale2 = 1.0 - m_lambda;                                

    // Compute the sum and the squared sum of the loss values             
    const double ept_sum = 
      std::accumulate(m_values.begin(), m_values.end(), 0.0);
    const double ept_sum_sq = 
      std::inner_product(m_values.begin(), m_values.end(), m_values.begin(), 0.0);

    const double var_sum = scale1 * ept_sum_sq + scale2 * ept_sum * ept_sum;

    // Compute the variational loss gradients (replace the expectation values)
    omp_set_num_threads(this->m_threads);
# pragma omp parallel for
    for (uint64_t s = 0; s < n_samples(); s ++)
    {
      for (uint64_t o = 0; o < n_outputs(); o ++)     
      {
        const double ept_val = m_values[s];
        const double ept_grd = m_grad(s, o);

        m_grad(s, o) = 2.0 * ept_grd * (scale1 * ept_val + scale2 * ept_sum);
      }
    }

    // Compute the variational loss values (replace the expectation values)
    std::fill(m_values.begin(), m_values.end(), var_sum * inverse(n_samples()));                
  }
  void LUTProblemVAR::update_loss(const Matrix<double>& scores)
  {
    // Compute the expectation loss values
    LUTProblemEPT::update_loss(scores);

    const double scale1 = m_lambda * n_samples();
    const double scale2 = 1.0 - m_lambda;                                

    // Compute the sum and the squared sum of the loss values             
    const double ept_sum = 
      std::accumulate(m_values.begin(), m_values.end(), 0.0);
    const double ept_sum_sq = 
      std::inner_product(m_values.begin(), m_values.end(), m_values.begin(), 0.0);

    const double var_sum = scale1 * ept_sum_sq + scale2 * ept_sum * ept_sum;

    // Compute the variational loss values (replace the expectation values)
    std::fill(m_values.begin(), m_values.end(), var_sum * inverse(n_samples()));
  }

}}
