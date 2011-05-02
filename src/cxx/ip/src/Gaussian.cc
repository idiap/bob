/**
 * @file src/cxx/ip/src/Gaussian.cc
 * @author <a href="mailto:Laurent.El-Shafey@idiap.ch">Laurent El Shafey</a> 
 *
 * @brief This file provides a class to process images with a Gaussian kernel
 */

#include "ip/Gaussian.h"

namespace ip = Torch::ip;

void ip::Gaussian::computeKernel()
{
  m_kernel.resize(2 * m_radius_y + 1, 2 * m_radius_x + 1);
  // compute the kernel
  const double inv_sigma = 1.0 / m_sigma;
  for (int j = -m_radius_y; j <= m_radius_y; j ++)
    for (int i = -m_radius_x; i <= m_radius_x; i++) {
      m_kernel(j + m_radius_y, i + m_radius_x) = 
        exp(- inv_sigma * (i * i + j * j));
    }

  // normalize the kernel
  m_kernel /= blitz::sum(m_kernel);
}

void ip::Gaussian::reset(const int radius_y, const int radius_x,
  const double sigma, const enum Torch::sp::Convolution::SizeOption size_opt,
  const enum Torch::sp::Convolution::BorderOption border_opt)
{
  m_radius_y = radius_y;
  m_radius_x = radius_x;
  m_sigma = sigma;
  m_conv_size = size_opt;
  m_conv_border = border_opt;
  computeKernel();
}
