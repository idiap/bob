/**
 * @file visioner/cxx/diag_exp_loss.cc
 * @date Fri 27 Jul 13:58:57 2012 CEST
 * @author Andre Anjos <andre.anjos@idiap.ch>
 *
 * @brief This file was part of Visioner and originally authored by "Cosmin
 * Atanasoaei <cosmin.atanasoaei@idiap.ch>". It was only modified to conform to
 * Bob coding standards and structure.
 *
 * Copyright (C) 2011-2013 Idiap Research Institute, Martigny, Switzerland
 */

#include "bob/visioner/model/losses/diag_exp_loss.h"

namespace bob { namespace visioner {

  // Compute the error (associated to the loss)
  double DiagExpLoss::error(double target, double score) const
  {
    return classification_error(target, score, 0.0);
  }

  // Compute the loss value & derivatives
  void DiagExpLoss::eval(
      double target, double score,
      double& value) const
  {
    const double eval = std::exp(- target * score);

    value = eval;
  }
  void DiagExpLoss::eval(
      double target, double score,
      double& value, double& deriv1) const
  {
    const double eval = std::exp(- target * score);

    value = eval;
    deriv1 = - target * eval;
  }
  void DiagExpLoss::eval(
      double target, double score,
      double& value, double& deriv1, double& deriv2) const
  {
    const double eval = std::exp(- target * score);

    value = eval;
    deriv1 = - target * eval;
    deriv2 = target * target * eval;
  }

}}
