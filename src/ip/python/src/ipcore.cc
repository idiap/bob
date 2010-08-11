/**
 * @file src/core/ipcore.cc
 * @author <a href="mailto:andre.anjos@idiap.ch">Andre Anjos</a> 
 *
 * @brief Binds the ipCore object type into python 
 */

#include <boost/python.hpp>

#include "core/spCore.h"
#include "core/ipCore.h"

using namespace boost::python;

void bind_core_ipcore()
{
  class_<Torch::ipCore, bases<Torch::spCore>, boost::noncopyable>("ipCore", "The base type for all image processing operators", no_init);
}
