/**
 * @file src/cxx/ip/src/Sobel.cc
 * @author <a href="mailto:Laurent.El-Shafey@idiap.ch">Laurent El Shafey</a> 
 *
 * @brief This file provides a class to process images with the Sobel operator
 */

#include "ip/Sobel.h"

namespace ip = Torch::ip;

ip::Sobel::Sobel( const bool up_positive, const bool left_positive,
    const enum sp::Convolution::SizeOption size_opt,
    const enum sp::Convolution::BorderOption border_opt):
  m_up_positive(up_positive), m_left_positive(left_positive),
  m_size_opt(size_opt), m_border_opt(border_opt)
{
  computeKernels();
}

void ip::Sobel::computeKernels()
{
  // Resize the kernels if required
  if( m_kernel_y.extent(0) != 3 || m_kernel_y.extent(1) != 3)
    m_kernel_y.resize(3,3);
  if( m_kernel_x.extent(0) != 3 || m_kernel_x.extent(1) != 3)
    m_kernel_x.resize(3,3);

  if(m_up_positive)
    m_kernel_y = 1, 2, 1, 0, 0, 0, -1, -2, -1;
  else
    m_kernel_y = -1, -2, -1, 0, 0, 0, 1, 2, 1;

  if(m_left_positive)
    m_kernel_x = 1, 0, -1, 2, 0, -1, 1, 0, -1;
  else
    m_kernel_x = -1, 0, 1, -2, 0, 1, -1, 0, 1;
}
