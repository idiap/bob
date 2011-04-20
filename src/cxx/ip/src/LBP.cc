/**
 * @file src/cxx/ip/src/LBP.cc
 * @author <a href="mailto:Laurent.El-Shafey@idiap.ch">Laurent El Shafey</a> 
 *
 * @brief LBP implementation
 */

#include "ip/LBP.h"

namespace ip = Torch::ip;

ip::LBP::LBP(const int P, const int R, const bool to_average,
    const bool add_average_bit, const bool uniform, 
    const bool rotation_invariant):
  m_P(P), m_R(R), m_to_average(to_average), m_add_average_bit(add_average_bit),
  m_uniform(uniform), m_rotation_invariant(rotation_invariant)
{
}
