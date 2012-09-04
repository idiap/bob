/**
 * @file visioner/visioner/model/losses/diag_symlog_loss.h
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

#ifndef BOB_VISIONER_DIAG_SYMLOG_LOSS_H
#define BOB_VISIONER_DIAG_SYMLOG_LOSS_H

#include "bob/visioner/model/losses/diag_loss.h"

namespace bob { namespace visioner {

  ////////////////////////////////////////////////////////////////////////////////
  // Symmetric logistic error loss: 
  //      l(y, f) = log(1 + exp(f - y)) + log(1 + exp(y - f)) - log(4.0),
  //              used by the diagonal multivariate loss.
  ////////////////////////////////////////////////////////////////////////////////

  class DiagSymLogLoss : public DiagLoss
  {
    public:

      // Constructor
      DiagSymLogLoss(const param_t& param = param_t())
        :       DiagLoss(param)
      {                        
      }       

      // Destructor
      virtual ~DiagSymLogLoss() {}

      // Reset to new parameters
      virtual void reset(const param_t& param) { m_param = param; }

      // Clone the object
      virtual boost::shared_ptr<Loss> clone() const { return boost::shared_ptr<Loss>(new DiagSymLogLoss(m_param)); }

    protected:

      // Compute the error (associated to the loss)
      virtual double error(double target, double score) const;

      // Compute the loss value & derivatives
      virtual void eval(
          double target, double score,
          double& value) const;
      virtual void eval(
          double target, double score,
          double& value, double& deriv1) const;
      virtual void eval(
          double target, double score,
          double& value, double& deriv1, double& deriv2) const;
  };

}}

#endif // DIAG_SYMLOG_LOSS_H
