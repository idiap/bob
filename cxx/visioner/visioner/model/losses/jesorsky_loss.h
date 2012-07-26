#ifndef BOB_VISIONER_JESORSKY_LOSS_H
#define BOB_VISIONER_JESORSKY_LOSS_H

#include "visioner/model/loss.h"

namespace bob { namespace visioner {

  /**
   * Diagonal multivariate losses: the sum (over outputs) of univariate losses.
   */
  class JesorskyLoss : public Loss {

    public:

      // Constructor
      JesorskyLoss(const param_t& param = param_t())
        :       Loss(param)
      {                        
      }

      // Destructor
      virtual ~JesorskyLoss() {}

      // Reset to new parameters
      virtual void reset(const param_t& param) { m_param = param; }

      // Clone the object
      virtual rloss_t clone() const { return rloss_t(new JesorskyLoss(m_param)); }

      // Compute the error (associated to the loss)
      virtual scalar_t error(
          const scalar_t* targets, const scalar_t* scores, index_t size) const;

      // Compute the loss value & derivatives
      virtual void eval(
          const scalar_t* targets, const scalar_t* scores, index_t size,
          scalar_t& value) const;
      virtual void eval(
          const scalar_t* targets, const scalar_t* scores, index_t size,
          scalar_t& value, scalar_t* grad) const;

    protected:

      // Compute the distance between the eyes
      scalar_t eye_dist(const scalar_t* targets) const
      {
        const scalar_t dx = targets[0] - targets[2];
        const scalar_t dy = targets[1] - targets[3];
        return my_sqrt(dx * dx + dy * dy);
      }

      // Compute the number of facial feature points
      index_t points(index_t size) const
      {
        return size >> 1;
      }
  };

}}

#endif // BOB_VISIONER_JESORSKY_LOSS_H
