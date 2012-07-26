#ifndef BOB_VISIONER_DIAG_SYMEXP_LOSS_H
#define BOB_VISIONER_DIAG_SYMEXP_LOSS_H

#include "visioner/model/losses/diag_loss.h"

namespace bob { namespace visioner {

  ////////////////////////////////////////////////////////////////////////////////
  // Symmetric exponential error loss:
  //      l(y, f) = exp(f - y) + exp(y - f) - 2.0,
  //              used by the diagonal multivariate loss.
  ////////////////////////////////////////////////////////////////////////////////

  class DiagSymExpLoss : public DiagLoss
  {
    public:

      // Constructor
      DiagSymExpLoss(const param_t& param = param_t())
        :       DiagLoss(param)
      {                        
      }       

      // Destructor
      virtual ~DiagSymExpLoss() {}

      // Reset to new parameters
      virtual void reset(const param_t& param) { m_param = param; }

      // Clone the object
      virtual rloss_t clone() const { return rloss_t(new DiagSymExpLoss(m_param)); }

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

#endif // BOB_VISIONER_DIAG_SYMEXP_LOSS_H
