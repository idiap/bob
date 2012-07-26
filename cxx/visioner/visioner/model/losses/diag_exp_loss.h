#ifndef BOB_VISIONER_DIAG_EXP_LOSS_H
#define BOB_VISIONER_DIAG_EXP_LOSS_H

#include "visioner/model/losses/diag_loss.h"

namespace bob { namespace visioner {

  ////////////////////////////////////////////////////////////////////////////////
  // Exponential univariate error loss:
  //      l(y, f) = exp(-y * f),
  //              used by the diagonal multivariate loss.
  ////////////////////////////////////////////////////////////////////////////////

  class DiagExpLoss : public DiagLoss
  {
    public:

      // Constructor
      DiagExpLoss(const param_t& param = param_t())
        :       DiagLoss(param)
      {                        
      }       

      // Destructor
      virtual ~DiagExpLoss() {}

      // Reset to new parameters
      virtual void reset(const param_t& param) { m_param = param; }

      // Clone the object
      virtual rloss_t clone() const { return rloss_t(new DiagExpLoss(m_param)); }

    protected:

      // Compute the error (associated to the loss)
      virtual scalar_t error(scalar_t target, scalar_t score) const;

      // Compute the loss value & derivatives
      virtual void eval(
          scalar_t target, scalar_t score,
          scalar_t& value) const;
      virtual void eval(
          scalar_t target, scalar_t score,
          scalar_t& value, scalar_t& deriv1) const;
      virtual void eval(
          scalar_t target, scalar_t score,
          scalar_t& value, scalar_t& deriv1, scalar_t& deriv2) const;
  };

}}

#endif // BOB_VISIONER_DIAG_EXP_LOSS_H
