/**
 * @file visioner/src/lut_problem_ept.cc
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

#include "visioner/util/threads.h"
#include "visioner/model/trainers/lutproblems/lut_problem_ept.h"

namespace bob { namespace visioner {

  // Constructor
  LUTProblemEPT::LUTProblemEPT(const DataSet& data, const param_t& param)
    :       LUTProblem(data, param),
    m_values(n_samples())
  {                
  }

  // Update loss values and derivatives
  void LUTProblemEPT::update_loss_deriv()
  {
    update_loss_deriv(m_sscores);
  }
  void LUTProblemEPT::update_loss()
  {
    update_loss(m_sscores);
  }

  // Update loss values and derivatives (for some particular scores)
  void LUTProblemEPT::update_loss_deriv(const Matrix<double>& scores)
  {
    // Allocate buffers (if not already done)
    m_grad.resize(n_samples(), n_outputs());

    // Compute the loss value + gradient
# pragma omp parallel for
    for (uint64_t s = 0; s < n_samples(); s ++)
    {
      m_loss.eval(target(s), scores[s], n_outputs(), m_values[s], m_grad[s]);

      // Adjust with costs
      const double _cost = cost(s);

      m_values[s] *= _cost;
      for (uint64_t o = 0; o < n_outputs(); o ++)
      {
        m_grad(s, o) *= _cost;
      }
    }                 
  }
  void LUTProblemEPT::update_loss(const Matrix<double>& scores)
  {
    // Compute the loss value
# pragma omp parallel for
    for (uint64_t s = 0; s < n_samples(); s ++)
    {
      m_loss.eval(target(s), scores[s], n_outputs(), m_values[s]);

      // Adjust with costs
      m_values[s] *= cost(s);
    }
  }

  // Compute the loss value/error
  double LUTProblemEPT::value() const
  {
    return  std::accumulate(m_values.begin(), m_values.end(), 0.0) * 
      inverse(n_samples()) * inverse(n_outputs());        
  }
  double LUTProblemEPT::error() const
  {
    double sum = 0.0;
    for (uint64_t s = 0; s < n_samples(); s ++)
    {
      sum += m_loss.error(target(s), m_sscores[s], n_outputs()) * cost(s);
    }

    return  sum *
      inverse(n_samples()) * inverse(n_outputs());
  }

  // Compute the gradient <g> and the function value in the <x> point
  //      (used during linesearch)
  double LUTProblemEPT::linesearch(const double* x, double* g)
  {
    update_cscores(x);
    update_loss_deriv(m_cscores);

    // Compute the loss value
    const double fx = std::accumulate(m_values.begin(), m_values.end(), 0.0);

    // Compute the gradients
    std::fill(g, g + n_outputs(), 0.0);
    for (uint64_t s = 0; s < n_samples(); s ++)
    {
      for (uint64_t o = 0; o < n_outputs(); o ++)
      {
        g[o] += m_grad(s, o) * m_wscores(s, o);
      }
    }

    return fx;
  }

  // Select the optimal feature
  void LUTProblemEPT::select()
  {
    // Allocate buffers (if not already done)
    m_fldeltas.resize(n_features(), n_outputs());
    m_fldeltas.fill(0.0);

    // Split the computation
    thread_loop(
        boost::bind(&LUTProblemEPT::select, this, boost::lambda::_1),
        n_features());

    // Decision: select the feature(s)
    switch (m_sharing)
    {
      // Independent features
      case Independent:
        {
          for (uint64_t o = 0; o < n_outputs(); o ++)
          {
            uint64_t bestf = 0;
            double besthv = 0.0;
            for (uint64_t f = 0; f < n_features(); f ++)
            {
              const double hv = m_fldeltas(f, o);
              if (hv < besthv)
              {
                bestf = f, besthv = hv;
              }
            }

            setup(bestf, o);
          }      
        }
        break;

        // Shared feature
      case Shared:
        {
          uint64_t bestf = 0;
          double besthv = 0.0;
          for (uint64_t f = 0; f < n_features(); f ++)
          {
            const double hv = 
              std::accumulate(m_fldeltas[f], m_fldeltas[f] + n_outputs(), 0.0);
            if (hv < besthv)
            {
              bestf = f, besthv = hv;
            }
          }

          for (uint64_t o = 0; o < n_outputs(); o ++)
          {
            setup(bestf, o);
          }
        }
        break;
    }
  }      

  // Compute the local loss decrease for a range of features
  void LUTProblemEPT::select(std::pair<uint64_t, uint64_t> frange)
  {
    // Evaluate each feature ...
    Matrix<double> histo_grad(n_entries(), n_outputs());                
    for (uint64_t f = frange.first; f < frange.second; f ++)
    {
      // - compute the loss gradient histogram
      histo(f, histo_grad);

      // - compute the local loss decrease
      for (uint64_t u = 0; u < n_entries(); u ++)
      {
        for (uint64_t o = 0; o < n_outputs(); o ++)
        {
          m_fldeltas(f, o) -= std::abs(histo_grad(u, o));
        }
      }
    }
  }

  // Compute the loss gradient histogram for a given feature
  void LUTProblemEPT::histo(uint64_t f, Matrix<double>& histo_grad) const
  {
    histo_grad.fill(0.0);
    for (uint64_t s = 0; s < n_samples(); s ++)
    {
      const uint16_t u = fvalue(f, s);
      for (uint64_t o = 0; o < n_outputs(); o ++)
      {
        histo_grad(u, o) += m_grad(s, o);
      }
    }
  }

  // Setup the given feature for the given output
  void LUTProblemEPT::setup(uint64_t f, uint64_t o)
  {
    Matrix<double> histo_grad(n_entries(), n_outputs());                
    histo(f, histo_grad);                

    // - set feature
    LUT& lut = m_luts[o];
    lut.feature() = f;

    // - set entries
    for (uint64_t u = 0; u < n_entries(); u ++)
    {
      lut[u] = histo_grad(u, o) > 0.0 ? -1.0 : 1.0;
      lut[u] *= m_umasks(f, u);
    }
  }

}}
