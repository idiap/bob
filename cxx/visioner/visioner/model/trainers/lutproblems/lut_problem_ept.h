/**
 * @file visioner/visioner/model/trainers/lutproblems/lut_problem_ept.h
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

#ifndef BOB_VISIONER_LUT_PROBLEM_EPT_H
#define BOB_VISIONER_LUT_PROBLEM_EPT_H

#include "visioner/model/trainers/lutproblems/lut_problem.h"

namespace bob { namespace visioner {

  ////////////////////////////////////////////////////////////////////////////////
  // LUTProblemEPT: 
  //      minimizes the cumulated expectation loss.
  ////////////////////////////////////////////////////////////////////////////////

  class LUTProblemEPT : public LUTProblem
  {
    public:

      // Constructor
      LUTProblemEPT(const DataSet& data, const param_t& param, size_t threads);

      // Destructor
      virtual ~LUTProblemEPT() {}

      // Update loss values and derivatives
      virtual void update_loss_deriv();
      virtual void update_loss();

      // Select the optimal feature
      virtual void select();

      // Optimize the LUT entries for the selected feature
      bool line_search();

      // Compute the loss value/error
      virtual double value() const;
      virtual double error() const;

      // Compute the gradient <g> and the function value in the <x> point
      //      (used during linesearch)
      virtual double linesearch(const double* x, double* g);

    protected:

      // Update loss values and derivatives (for some particular scores)
      virtual void update_loss_deriv(const Matrix<double>& scores);
      virtual void update_loss(const Matrix<double>& scores);

      // Compute the local loss decrease for a range of features
      void select(std::pair<uint64_t, uint64_t> frange);

      // Compute the loss gradient histogram for a given feature
      void histo(uint64_t f, Matrix<double>& histo) const;

      // Setup the given feature for the given output
      void setup(uint64_t f, uint64_t o);


    private: //multi-threading

      void update_loss_deriv_mt(const Matrix<double>& scores, 
          const std::pair<uint64_t,uint64_t>& range);

      void update_loss_mt(const Matrix<double>& scores, 
          const std::pair<uint64_t,uint64_t>& range);

    protected:

      // Attributes
      std::vector<double>               m_values;       // Loss values
      Matrix<double>            m_grad;         // Loss gradients

      Matrix<double>            m_fldeltas;     // (feature, output) -> local loss decrease

  };	

}}

#endif // BOB_VISIONER_LUT_PROBLEM_EPT_H
