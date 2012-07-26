#include <numeric>

#include "visioner/model/trainers/lutproblems/lut_problem_var.h"

namespace bob { namespace visioner {

  // Constructor
  LUTProblemVAR::LUTProblemVAR(const DataSet& data, const param_t& param, scalar_t lambda)
    :       LUTProblemEPT(data, param), 
    m_lambda(lambda)
  {                
  }

  // Update loss values and derivatives (for some particular scores)
  void LUTProblemVAR::update_loss_deriv(const scalar_mat_t& scores)
  {
    // Compute the expectation loss values and derivatives
    LUTProblemEPT::update_loss_deriv(scores);

    const scalar_t scale1 = m_lambda * n_samples();
    const scalar_t scale2 = 1.0 - m_lambda;                                

    // Compute the sum and the squared sum of the loss values             
    const scalar_t ept_sum = 
      std::accumulate(m_values.begin(), m_values.end(), 0.0);
    const scalar_t ept_sum_sq = 
      std::inner_product(m_values.begin(), m_values.end(), m_values.begin(), 0.0);

    const scalar_t var_sum = scale1 * ept_sum_sq + scale2 * ept_sum * ept_sum;

    // Compute the variational loss gradients (replace the expectation values)
# pragma omp parallel for
    for (index_t s = 0; s < n_samples(); s ++)
    {
      for (index_t o = 0; o < n_outputs(); o ++)     
      {
        const scalar_t ept_val = m_values[s];
        const scalar_t ept_grd = m_grad(s, o);

        m_grad(s, o) = 2.0 * ept_grd * (scale1 * ept_val + scale2 * ept_sum);
      }
    }

    // Compute the variational loss values (replace the expectation values)
    std::fill(m_values.begin(), m_values.end(), var_sum * inverse(n_samples()));                
  }
  void LUTProblemVAR::update_loss(const scalar_mat_t& scores)
  {
    // Compute the expectation loss values
    LUTProblemEPT::update_loss(scores);

    const scalar_t scale1 = m_lambda * n_samples();
    const scalar_t scale2 = 1.0 - m_lambda;                                

    // Compute the sum and the squared sum of the loss values             
    const scalar_t ept_sum = 
      std::accumulate(m_values.begin(), m_values.end(), 0.0);
    const scalar_t ept_sum_sq = 
      std::inner_product(m_values.begin(), m_values.end(), m_values.begin(), 0.0);

    const scalar_t var_sum = scale1 * ept_sum_sq + scale2 * ept_sum * ept_sum;

    // Compute the variational loss values (replace the expectation values)
    std::fill(m_values.begin(), m_values.end(), var_sum * inverse(n_samples()));
  }

}}
