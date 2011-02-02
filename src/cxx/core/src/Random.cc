/**
 * @file src/cxx/core/src/Random.cc
 * @author <a href="mailto:Laurent.El-Shafey@idiap.ch">Laurent El Shafey</a>
 * @author <a href="mailto:andre.anjos@idiap.ch">Andre Anjos</a>
 *
 * @brief Random integer generators based on Boost
 */

#include "core/Random.h"

boost::mt19937& 
Torch::core::random::generator::instance()
{
  if( !s_generator)
    s_generator.reset(new boost::mt19937);
  return *s_generator.get();
}

boost::scoped_ptr<boost::mt19937> Torch::core::random::generator::s_generator;

