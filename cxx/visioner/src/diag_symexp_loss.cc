#include "visioner/model/losses/diag_symexp_loss.h"

namespace bob { namespace visioner {

  // Compute the error (associated to the loss)
  scalar_t DiagSymExpLoss::error(scalar_t target, scalar_t score) const
  {
    return regression_error(target, score, 0.0);
  }

  // Compute the loss value & derivatives
  void DiagSymExpLoss::eval(
      scalar_t target, scalar_t score,
      scalar_t& value) const
  {
    static const scalar_t delta = 2.0;

    const scalar_t eval = my_exp(score - target);
    const scalar_t ieval = 1.0 / eval;

    value = eval + ieval - delta;
  }
  void DiagSymExpLoss::eval(
      scalar_t target, scalar_t score,
      scalar_t& value, scalar_t& deriv1) const
  {
    static const scalar_t delta = 2.0;

    const scalar_t eval = my_exp(score - target);
    const scalar_t ieval = 1.0 / eval;

    value = eval + ieval - delta;
    deriv1 = eval - ieval;
  }
  void DiagSymExpLoss::eval(
      scalar_t target, scalar_t score,
      scalar_t& value, scalar_t& deriv1, scalar_t& deriv2) const
  {
    static const scalar_t delta = 2.0;

    const scalar_t eval = my_exp(score - target);
    const scalar_t ieval = 1.0 / eval;

    value = eval + ieval - delta;
    deriv1 = eval - ieval;
    deriv2 = eval + ieval;
  }

}}
