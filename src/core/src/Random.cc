#include "core/Random.hpp"

Torch::core::random::generator::~generator()
{
}

boost::mt19937& 
Torch::core::random::generator::instance()
{
  if( !s_generator)
    s_generator.reset(new boost::mt19937);
  return *s_generator.get();
}

boost::scoped_ptr<boost::mt19937> Torch::core::random::generator::s_generator;
   
