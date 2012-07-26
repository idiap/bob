#include "visioner/model/losses/diag_symlog_loss.h"

namespace bob { namespace visioner {

  // Compute the error (associated to the loss)
  scalar_t DiagSymLogLoss::error(scalar_t target, scalar_t score) const
  {
    return regression_error(target, score, 0.0);
  }

  // Compute the loss value & derivatives
  void DiagSymLogLoss::eval(
      scalar_t target, scalar_t score,
      scalar_t& value) const
  {
    static const scalar_t delta = my_log(4.0);

    const scalar_t eval = my_exp(score - target);

    value = my_log(2.0 + eval + 1.0 / eval) - delta;
  }
  void DiagSymLogLoss::eval(
      scalar_t target, scalar_t score,
      scalar_t& value, scalar_t& deriv1) const
  {
    static const scalar_t delta = my_log(4.0);

    const scalar_t eval = my_exp(score - target);
    const scalar_t norm = 1.0 / (1.0 + eval);

    value = my_log(2.0 + eval + 1.0 / eval) - delta;
    deriv1 = (eval - 1) * norm;
  }
  void DiagSymLogLoss::eval(
      scalar_t target, scalar_t score,
      scalar_t& value, scalar_t& deriv1, scalar_t& deriv2) const
  {
    static const scalar_t delta = my_log(4.0);

    const scalar_t eval = my_exp(score - target);
    const scalar_t norm = 1.0 / (1.0 + eval);

    value = my_log(2.0 + eval + 1.0 / eval) - delta;
    deriv1 = (eval - 1) * norm;
    deriv2 = 2.0 * eval * norm * norm;
  }

}}
