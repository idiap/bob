/**
 * @file bob/visioner/model/losses/diag_loss.h
 * @date Fri 27 Jul 13:58:57 2012 CEST
 * @author Andre Anjos <andre.anjos@idiap.ch>
 *
 * @brief This file was part of Visioner and originally authored by "Cosmin
 * Atanasoaei <cosmin.atanasoaei@idiap.ch>". It was only modified to conform to
 * Bob coding standards and structure.
 *
 * Copyright (C) 2011-2013 Idiap Research Institute, Martigny, Switzerland
 */

#ifndef BOB_VISIONER_DIAG_LOSS_H
#define BOB_VISIONER_DIAG_LOSS_H

#include "bob/visioner/model/loss.h"

namespace bob { namespace visioner {

  ////////////////////////////////////////////////////////////////////////////////
  // Diagonal multivariate losses: 
  //      the sum (over outputs) of univariate losses.
  ////////////////////////////////////////////////////////////////////////////////

  class DiagLoss : public Loss
  {
    public:

      // Constructor
      DiagLoss(const param_t& param = param_t())
        :       Loss(param)
      {                        
      }

      // Destructor
      virtual ~DiagLoss() {}

      // Compute the error (associated to the loss)
      virtual double error(
          const double* targets, const double* scores, uint64_t size) const;

      // Compute the loss value & derivatives
      virtual void eval(
          const double* targets, const double* scores, uint64_t size,
          double& value) const;
      virtual void eval(
          const double* targets, const double* scores, uint64_t size,
          double& value, double* grad) const;

    protected:

      // Compute the error (associated to the loss)
      virtual double error(double target, double score) const = 0;

      // Compute the loss value & derivatives
      virtual void eval(
          double target, double score,
          double& value) const = 0;
      virtual void eval(
          double target, double score,
          double& value, double& deriv1) const = 0;
  };

}}

#endif // BOB_VISIONER_DIAG_LOSS_H
