#include "visioner/model/losses/diag_log_loss.h"

namespace bob { namespace visioner {

  // Compute the error (associated to the loss)
  scalar_t DiagLogLoss::error(scalar_t target, scalar_t score) const
  {
    return classification_error(target, score, 0.0);
  }

  // Compute the loss value & derivatives
  void DiagLogLoss::eval(
      scalar_t target, scalar_t score,
      scalar_t& value) const
  {
    const scalar_t eval = my_exp(- target * score);

    value = my_log(1.0 + eval);
  }
  void DiagLogLoss::eval(
      scalar_t target, scalar_t score,
      scalar_t& value, scalar_t& deriv1) const
  {
    const scalar_t eval = my_exp(- target * score);
    const scalar_t norm = 1.0 / (1.0 + eval);

    value = my_log(1.0 + eval);
    deriv1 = - target * eval * norm;
  }
  void DiagLogLoss::eval(
      scalar_t target, scalar_t score,
      scalar_t& value, scalar_t& deriv1, scalar_t& deriv2) const
  {
    const scalar_t eval = my_exp(- target * score);
    const scalar_t norm = 1.0 / (1.0 + eval);

    value = my_log(1.0 + eval);
    deriv1 = - target * eval * norm;
    deriv2 = target * target * eval * norm * norm;
  }

}}
