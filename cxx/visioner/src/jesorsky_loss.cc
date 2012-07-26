#include "visioner/model/losses/jesorsky_loss.h"

namespace bob { namespace visioner {

  // Compute the error (associated to the loss)
  scalar_t JesorskyLoss::error(
      const scalar_t* targets, const scalar_t* scores, index_t size) const
  {
    const index_t n_points = points(size);
    const scalar_t scale = 2.0 * inverse(eye_dist(targets));

    scalar_t sum_error = 0;           
    for (index_t p = 0; p < n_points; p ++)
    {
      const scalar_t dx = scores[2 * p + 0] - targets[2 * p + 0];
      const scalar_t dy = scores[2 * p + 1] - targets[2 * p + 1];
      sum_error += scale * my_sqrt(dx * dx + dy * dy);
    }

    return sum_error;
  }

  // Compute the loss value & derivatives
  void JesorskyLoss::eval(
      const scalar_t* targets, const scalar_t* scores, index_t size,
      scalar_t& value) const
  {
    value = 0.0;

    const index_t n_points = points(size);
    const scalar_t scale = 2.0 * inverse(eye_dist(targets));

    for (index_t p = 0; p < n_points; p ++)
    {
      const scalar_t dx = scores[2 * p + 0] - targets[2 * p + 0];
      const scalar_t dy = scores[2 * p + 1] - targets[2 * p + 1];
      value += scale * my_sqrt(dx * dx + dy * dy);
    }
  }
  void JesorskyLoss::eval(
      const scalar_t* targets, const scalar_t* scores, index_t size,
      scalar_t& value, scalar_t* grad) const
  {
    value = 0.0;

    const index_t n_points = points(size);
    const scalar_t scale = 2.0 * inverse(eye_dist(targets));

    for (index_t p = 0; p < n_points; p ++)
    {
      const scalar_t dx = scores[2 * p + 0] - targets[2 * p + 0];
      const scalar_t dy = scores[2 * p + 1] - targets[2 * p + 1];

      const scalar_t sq = my_sqrt(dx * dx + dy * dy);
      const scalar_t isq = inverse(sq);

      value += scale * sq;
      grad[2 * p + 0] = scale * isq * dx;
      grad[2 * p + 1] = scale * isq * dy;
    }
  }

}}
