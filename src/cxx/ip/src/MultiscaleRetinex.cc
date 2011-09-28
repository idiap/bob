/**
 * @file src/cxx/ip/src/MultiscaleRetinex.cc
 * @author <a href="mailto:Laurent.El-Shafey@idiap.ch">Laurent El Shafey</a> 
 *
 * @brief Implements the MultiscaleRetinex algorithm
 */

#include "ip/MultiscaleRetinex.h"

namespace ip = Torch::ip;

void ip::MultiscaleRetinex::computeKernels()
{
  for( size_t s=0; s<m_n_scales; ++s)
  {
    // size of the kernel 
    int s_size = m_size_min + s * m_size_step;
    // sigma of the kernel
    double s_sigma = m_sigma * s_size / m_size_min;
    // Initialize the Gaussian
    m_gaussians[s].reset(s_size, s_size, s_sigma, s_sigma, 
      Torch::sp::Convolution::Same, m_conv_border);
  }
}
