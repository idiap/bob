/**
 * @file src/cxx/ip/src/Rotate.cc
 * @author <a href="mailto:Laurent.El-Shafey@idiap.ch">Laurent El Shafey</a> 
 *
 * @brief This file provides a class to perform a rotation
 */

#include "ip/Rotate.h"

namespace ip = Torch::ip;

ip::Rotate::Rotate( const double angle, 
  const ip::Rotate::Algorithm algo):
  m_angle(angle), m_algo(algo)
{
} 

ip::Rotate::~Rotate() { }

