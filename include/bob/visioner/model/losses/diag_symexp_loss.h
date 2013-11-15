/**
 * @file bob/visioner/model/losses/diag_symexp_loss.h
 * @date Fri 27 Jul 13:58:57 2012 CEST
 * @author Andre Anjos <andre.anjos@idiap.ch>
 *
 * @brief This file was part of Visioner and originally authored by "Cosmin
 * Atanasoaei <cosmin.atanasoaei@idiap.ch>". It was only modified to conform to
 * Bob coding standards and structure.
 *
 * Copyright (C) 2011-2013 Idiap Research Institute, Martigny, Switzerland
 */

#ifndef BOB_VISIONER_DIAG_SYMEXP_LOSS_H
#define BOB_VISIONER_DIAG_SYMEXP_LOSS_H

#include "bob/visioner/model/losses/diag_loss.h"

namespace bob { namespace visioner {

  ////////////////////////////////////////////////////////////////////////////////
  // Symmetric exponential error loss:
  //      l(y, f) = exp(f - y) + exp(y - f) - 2.0,
  //              used by the diagonal multivariate loss.
  ////////////////////////////////////////////////////////////////////////////////

  class DiagSymExpLoss : public DiagLoss
  {
    public:

      // Constructor
      DiagSymExpLoss(const param_t& param = param_t())
        :       DiagLoss(param)
      {                        
      }       

      // Destructor
      virtual ~DiagSymExpLoss() {}

      // Reset to new parameters
      virtual void reset(const param_t& param) { m_param = param; }

      // Clone the object
      virtual boost::shared_ptr<Loss> clone() const { return boost::shared_ptr<Loss>(new DiagSymExpLoss(m_param)); }

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

#endif // BOB_VISIONER_DIAG_SYMEXP_LOSS_H
