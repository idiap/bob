#include "visioner/model/losses/diag_loss.h"

namespace bob { namespace visioner {

  // Compute the error (associated to the loss)
  scalar_t DiagLoss::error(
      const scalar_t* targets, const scalar_t* scores, index_t size) const
  {
    scalar_t sum_error = 0;

    for (index_t o = 0; o < size; o ++)
    {
      sum_error += error(targets[o], scores[o]);
    }

    return sum_error;
  }

  // Compute the loss value & derivatives
  void DiagLoss::eval(
      const scalar_t* targets, const scalar_t* scores, index_t size,
      scalar_t& value) const
  {
    value = 0.0;

    for (index_t o = 0; o < size; o ++)
    {
      scalar_t ovalue;
      eval(targets[o], scores[o], ovalue);

      value += ovalue;
    }
  }
  void DiagLoss::eval(
      const scalar_t* targets, const scalar_t* scores, index_t size,
      scalar_t& value, scalar_t* grad) const
  {
    value = 0.0;                
    for (index_t o = 0; o < size; o ++)
    {
      scalar_t ovalue, oderiv1;
      eval(targets[o], scores[o], ovalue, oderiv1);

      value += ovalue;
      grad[o] = oderiv1;
    }
  }

}}
