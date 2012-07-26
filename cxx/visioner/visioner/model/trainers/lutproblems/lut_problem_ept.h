#ifndef BOB_VISIONER_LUT_PROBLEM_EPT_H
#define BOB_VISIONER_LUT_PROBLEM_EPT_H

#include "visioner/model/trainers/lutproblems/lut_problem.h"

namespace bob { namespace visioner {

  ////////////////////////////////////////////////////////////////////////////////
  // LUTProblemEPT: 
  //      minimizes the cumulated expectation loss.
  ////////////////////////////////////////////////////////////////////////////////

  class LUTProblemEPT : public LUTProblem
  {
    public:

      // Constructor
      LUTProblemEPT(const DataSet& data, const param_t& param);

      // Destructor
      virtual ~LUTProblemEPT() {}

      // Update loss values and derivatives
      virtual void update_loss_deriv();
      virtual void update_loss();

      // Select the optimal feature
      virtual void select();

      // Optimize the LUT entries for the selected feature
      bool line_search();

      // Compute the loss value/error
      virtual scalar_t value() const;
      virtual scalar_t error() const;

      // Compute the gradient <g> and the function value in the <x> point
      //      (used during linesearch)
      virtual scalar_t linesearch(const scalar_t* x, scalar_t* g);

    protected:

      // Update loss values and derivatives (for some particular scores)
      virtual void update_loss_deriv(const scalar_mat_t& scores);
      virtual void update_loss(const scalar_mat_t& scores);

      // Compute the local loss decrease for a range of features
      void select(index_pair_t frange);

      // Compute the loss gradient histogram for a given feature
      void histo(index_t f, scalar_mat_t& histo) const;

      // Setup the given feature for the given output
      void setup(index_t f, index_t o);

    protected:

      // Attributes
      scalars_t               m_values;       // Loss values
      scalar_mat_t            m_grad;         // Loss gradients

      scalar_mat_t            m_fldeltas;     // (feature, output) -> local loss decrease
  };	

}}

#endif // BOB_VISIONER_LUT_PROBLEM_EPT_H
