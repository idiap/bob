#ifndef BOB_VISIONER_LOSS_H
#define BOB_VISIONER_LOSS_H

#include "visioner/model/param.h"

namespace bob { namespace visioner {

  class Loss;
  typedef boost::shared_ptr<Loss>		rloss_t;

  ////////////////////////////////////////////////////////////////////////////////
  // Generic multivariate loss function of two parameters:
  //	the target value to predict and the current score estimation.
  //
  // NB: The <grad> and <hess> objects should be a-priori
  //      resized to <size, 1> and <size, size> respectively, 
  //      before calling <::eval>.
  ////////////////////////////////////////////////////////////////////////////////

  class Loss : public Parametrizable
  {
    public:

      // Constructor
      Loss(const param_t& param = param_t())
        :       Parametrizable(param)
      {                        
      }

      // Destructor
      virtual ~Loss() {}

      // Clone the object
      virtual rloss_t clone() const = 0;

      // Compute the error (associated to the loss)
      virtual scalar_t error(
          const scalar_t* targets, const scalar_t* scores, index_t size) const = 0;

      // Compute the loss value & derivatives
      virtual void eval(
          const scalar_t* targets, const scalar_t* scores, index_t size,
          scalar_t& value) const = 0;
      virtual void eval(
          const scalar_t* targets, const scalar_t* scores, index_t size,
          scalar_t& value, scalar_t* grad) const = 0;
  };

}}

#endif // BOB_VISIONER_LOSS_H
