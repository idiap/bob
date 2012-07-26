#ifndef BOB_VISIONER_LUT_PROBLEM_VAR_H
#define BOB_VISIONER_LUT_PROBLEM_VAR_H

#include "visioner/model/trainers/lutproblems/lut_problem_ept.h" 

namespace bob { namespace visioner {

  ////////////////////////////////////////////////////////////////////////////////
  // LUTProblemVAR: 
  //      minimizes the cumulated expectation loss
  //      regularized with the loss variance.
  ////////////////////////////////////////////////////////////////////////////////

  class LUTProblemVAR : public LUTProblemEPT
  {
    public:

      // Constructor
      LUTProblemVAR(const DataSet& data, const param_t& param, scalar_t lambda);

      // Destructor
      virtual ~LUTProblemVAR() {} 

    protected:

      // Update loss values and derivatives (for some particular scores)
      virtual void update_loss_deriv(const scalar_mat_t& scores);
      virtual void update_loss(const scalar_mat_t& scores);

    protected:

      // Attributes
      scalar_t                        m_lambda;       // Regularization factor
  };	

}}

#endif // BOB_VISIONER_LUT_PROBLEM_VAR_H
