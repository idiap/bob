/**
 * @file bob/visioner/model/losses/diag_exp_loss.h
 * @date Fri 27 Jul 13:58:57 2012 CEST
 * @author Andre Anjos <andre.anjos@idiap.ch>
 *
 * @brief This file was part of Visioner and originally authored by "Cosmin
 * Atanasoaei <cosmin.atanasoaei@idiap.ch>". It was only modified to conform to
 * Bob coding standards and structure.
 *
 * Copyright (C) 2011-2013 Idiap Research Institute, Martigny, Switzerland
 */

#ifndef BOB_VISIONER_DIAG_EXP_LOSS_H
#define BOB_VISIONER_DIAG_EXP_LOSS_H

#include "bob/visioner/model/losses/diag_loss.h"

namespace bob { namespace visioner {

  ////////////////////////////////////////////////////////////////////////////////
  // Exponential univariate error loss:
  //      l(y, f) = exp(-y * f),
  //              used by the diagonal multivariate loss.
  ////////////////////////////////////////////////////////////////////////////////

  class DiagExpLoss : public DiagLoss
  {
    public:

      // Constructor
      DiagExpLoss(const param_t& param = param_t())
        :       DiagLoss(param)
      {                        
      }       

      // Destructor
      virtual ~DiagExpLoss() {}

      // Reset to new parameters
      virtual void reset(const param_t& param) { m_param = param; }

      // Clone the object
      virtual boost::shared_ptr<Loss> clone() const { return boost::shared_ptr<Loss>(new DiagExpLoss(m_param)); }

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

#endif // BOB_VISIONER_DIAG_EXP_LOSS_H
