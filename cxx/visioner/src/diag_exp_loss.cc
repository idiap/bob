#include "visioner/model/losses/diag_exp_loss.h"

namespace bob { namespace visioner {

  // Compute the error (associated to the loss)
  scalar_t DiagExpLoss::error(scalar_t target, scalar_t score) const
  {
    return classification_error(target, score, 0.0);
  }

  // Compute the loss value & derivatives
  void DiagExpLoss::eval(
      scalar_t target, scalar_t score,
      scalar_t& value) const
  {
    const scalar_t eval = my_exp(- target * score);

    value = eval;
  }
  void DiagExpLoss::eval(
      scalar_t target, scalar_t score,
      scalar_t& value, scalar_t& deriv1) const
  {
    const scalar_t eval = my_exp(- target * score);

    value = eval;
    deriv1 = - target * eval;
  }
  void DiagExpLoss::eval(
      scalar_t target, scalar_t score,
      scalar_t& value, scalar_t& deriv1, scalar_t& deriv2) const
  {
    const scalar_t eval = my_exp(- target * score);

    value = eval;
    deriv1 = - target * eval;
    deriv2 = target * target * eval;
  }

}}
