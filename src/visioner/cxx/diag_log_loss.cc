/**
 * @file visioner/cxx/diag_log_loss.cc
 * @date Fri 27 Jul 13:58:57 2012 CEST
 * @author Andre Anjos <andre.anjos@idiap.ch>
 *
 * @brief This file was part of Visioner and originally authored by "Cosmin
 * Atanasoaei <cosmin.atanasoaei@idiap.ch>". It was only modified to conform to
 * Bob coding standards and structure.
 *
 * Copyright (C) 2011-2013 Idiap Research Institute, Martigny, Switzerland
 */

#include "bob/visioner/model/losses/diag_log_loss.h"

namespace bob { namespace visioner {

  // Compute the error (associated to the loss)
  double DiagLogLoss::error(double target, double score) const
  {
    return classification_error(target, score, 0.0);
  }

  // Compute the loss value & derivatives
  void DiagLogLoss::eval(
      double target, double score,
      double& value) const
  {
    const double eval = std::exp(- target * score);

    value = std::log(1.0 + eval);
  }
  void DiagLogLoss::eval(
      double target, double score,
      double& value, double& deriv1) const
  {
    const double eval = std::exp(- target * score);
    const double norm = 1.0 / (1.0 + eval);

    value = std::log(1.0 + eval);
    deriv1 = - target * eval * norm;
  }
  void DiagLogLoss::eval(
      double target, double score,
      double& value, double& deriv1, double& deriv2) const
  {
    const double eval = std::exp(- target * score);
    const double norm = 1.0 / (1.0 + eval);

    value = std::log(1.0 + eval);
    deriv1 = - target * eval * norm;
    deriv2 = target * target * eval * norm * norm;
  }

}}
