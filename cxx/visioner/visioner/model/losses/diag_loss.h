#ifndef BOB_VISIONER_DIAG_LOSS_H
#define BOB_VISIONER_DIAG_LOSS_H

#include "visioner/model/loss.h"

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

      // Compute the error (associated to the loss)
      virtual scalar_t error(scalar_t target, scalar_t score) const = 0;

      // Compute the loss value & derivatives
      virtual void eval(
          scalar_t target, scalar_t score,
          scalar_t& value) const = 0;
      virtual void eval(
          scalar_t target, scalar_t score,
          scalar_t& value, scalar_t& deriv1) const = 0;
  };

}}

#endif // BOB_VISIONER_DIAG_LOSS_H
