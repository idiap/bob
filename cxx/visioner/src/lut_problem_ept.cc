#include <numeric>

#include "visioner/model/trainers/lutproblems/lut_problem_ept.h"

namespace bob { namespace visioner {

  // Constructor
  LUTProblemEPT::LUTProblemEPT(const DataSet& data, const param_t& param)
    :       LUTProblem(data, param),
    m_values(n_samples())
  {                
  }

  // Update loss values and derivatives
  void LUTProblemEPT::update_loss_deriv()
  {
    update_loss_deriv(m_sscores);
  }
  void LUTProblemEPT::update_loss()
  {
    update_loss(m_sscores);
  }

  // Update loss values and derivatives (for some particular scores)
  void LUTProblemEPT::update_loss_deriv(const scalar_mat_t& scores)
  {
    // Allocate buffers (if not already done)
    m_grad.resize(n_samples(), n_outputs());

    // Compute the loss value + gradient
# pragma omp parallel for
    for (index_t s = 0; s < n_samples(); s ++)
    {
      m_loss.eval(target(s), scores[s], n_outputs(), m_values[s], m_grad[s]);

      // Adjust with costs
      const scalar_t _cost = cost(s);

      m_values[s] *= _cost;
      for (index_t o = 0; o < n_outputs(); o ++)
      {
        m_grad(s, o) *= _cost;
      }
    }                 
  }
  void LUTProblemEPT::update_loss(const scalar_mat_t& scores)
  {
    // Compute the loss value
# pragma omp parallel for
    for (index_t s = 0; s < n_samples(); s ++)
    {
      m_loss.eval(target(s), scores[s], n_outputs(), m_values[s]);

      // Adjust with costs
      m_values[s] *= cost(s);
    }
  }

  // Compute the loss value/error
  scalar_t LUTProblemEPT::value() const
  {
    return  std::accumulate(m_values.begin(), m_values.end(), 0.0) * 
      inverse(n_samples()) * inverse(n_outputs());        
  }
  scalar_t LUTProblemEPT::error() const
  {
    scalar_t sum = 0.0;
    for (index_t s = 0; s < n_samples(); s ++)
    {
      sum += m_loss.error(target(s), m_sscores[s], n_outputs()) * cost(s);
    }

    return  sum *
      inverse(n_samples()) * inverse(n_outputs());
  }

  // Compute the gradient <g> and the function value in the <x> point
  //      (used during linesearch)
  scalar_t LUTProblemEPT::linesearch(const scalar_t* x, scalar_t* g)
  {
    update_cscores(x);
    update_loss_deriv(m_cscores);

    // Compute the loss value
    const scalar_t fx = std::accumulate(m_values.begin(), m_values.end(), 0.0);

    // Compute the gradients
    std::fill(g, g + n_outputs(), 0.0);
    for (index_t s = 0; s < n_samples(); s ++)
    {
      for (index_t o = 0; o < n_outputs(); o ++)
      {
        g[o] += m_grad(s, o) * m_wscores(s, o);
      }
    }

    return fx;
  }

  // Select the optimal feature
  void LUTProblemEPT::select()
  {
    // Allocate buffers (if not already done)
    m_fldeltas.resize(n_features(), n_outputs());
    m_fldeltas.fill(0.0);

    // Split the computation
    thread_loop(
        boost::bind(&LUTProblemEPT::select, this, boost::lambda::_1),
        n_features());

    // Decision: select the feature(s)
    switch (m_sharing)
    {
      // Independent features
      case Independent:
        {
          for (index_t o = 0; o < n_outputs(); o ++)
          {
            index_t bestf = 0;
            scalar_t besthv = 0.0;
            for (index_t f = 0; f < n_features(); f ++)
            {
              const scalar_t hv = m_fldeltas(f, o);
              if (hv < besthv)
              {
                bestf = f, besthv = hv;
              }
            }

            setup(bestf, o);
          }      
        }
        break;

        // Shared feature
      case Shared:
        {
          index_t bestf = 0;
          scalar_t besthv = 0.0;
          for (index_t f = 0; f < n_features(); f ++)
          {
            const scalar_t hv = 
              std::accumulate(m_fldeltas[f], m_fldeltas[f] + n_outputs(), 0.0);
            if (hv < besthv)
            {
              bestf = f, besthv = hv;
            }
          }

          for (index_t o = 0; o < n_outputs(); o ++)
          {
            setup(bestf, o);
          }
        }
        break;
    }
  }      

  // Compute the local loss decrease for a range of features
  void LUTProblemEPT::select(index_pair_t frange)
  {
    // Evaluate each feature ...
    scalar_mat_t histo_grad(n_entries(), n_outputs());                
    for (index_t f = frange.first; f < frange.second; f ++)
    {
      // - compute the loss gradient histogram
      histo(f, histo_grad);

      // - compute the local loss decrease
      for (index_t u = 0; u < n_entries(); u ++)
      {
        for (index_t o = 0; o < n_outputs(); o ++)
        {
          m_fldeltas(f, o) -= my_abs(histo_grad(u, o));
        }
      }
    }
  }

  // Compute the loss gradient histogram for a given feature
  void LUTProblemEPT::histo(index_t f, scalar_mat_t& histo_grad) const
  {
    histo_grad.fill(0.0);
    for (index_t s = 0; s < n_samples(); s ++)
    {
      const discrete_t u = fvalue(f, s);
      for (index_t o = 0; o < n_outputs(); o ++)
      {
        histo_grad(u, o) += m_grad(s, o);
      }
    }
  }

  // Setup the given feature for the given output
  void LUTProblemEPT::setup(index_t f, index_t o)
  {
    scalar_mat_t histo_grad(n_entries(), n_outputs());                
    histo(f, histo_grad);                

    // - set feature
    LUT& lut = m_luts[o];
    lut.feature() = f;

    // - set entries
    for (index_t u = 0; u < n_entries(); u ++)
    {
      lut[u] = histo_grad(u, o) > 0.0 ? -1.0 : 1.0;
      lut[u] *= m_umasks(f, u);
    }
  }

}}
