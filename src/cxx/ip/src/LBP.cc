/**
 * @file src/cxx/ip/src/LBP.cc
 * @author <a href="mailto:Laurent.El-Shafey@idiap.ch">Laurent El Shafey</a> 
 *
 * @brief LBP implementation
 */

#include "ip/LBP.h"

namespace ip = Torch::ip;

ip::LBP::LBP(const int P, const double R, const bool circular,
    const bool to_average,const bool add_average_bit, const bool uniform,
    const bool rotation_invariant):
  m_P(P), m_R(R), m_circular(circular), m_to_average(to_average), 
  m_add_average_bit(add_average_bit), m_uniform(uniform), 
  m_rotation_invariant(rotation_invariant),
  m_lut_RI(0), m_lut_U2(0), m_lut_U2RI(0), m_lut_add_average_bit(0), 
  m_lut_normal(0), m_lut_current(0)
{
  updateR(); 
}

void ip::LBP::init_lut_current()
{
  // Reference to the current lookup table
  if(m_rotation_invariant)
  {
    if(m_uniform)
      m_lut_current.reference( m_lut_U2RI );
    else
      m_lut_current.reference( m_lut_RI );
  }
  else
  {
    if(m_uniform)
      m_lut_current.reference( m_lut_U2 );
    else
    {
      if(m_add_average_bit && m_to_average)
        m_lut_current.reference( m_lut_add_average_bit );
      else
        m_lut_current.reference( m_lut_normal );
    }   
  }
}
