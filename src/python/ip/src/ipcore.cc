/**
 * @file src/ip/python/src/ipcore.cc
 * @author <a href="mailto:andre.anjos@idiap.ch">Andre Anjos</a> 
 *
 * @brief Binds the ipCore object type into python 
 */

#include <boost/python.hpp>

#include "sp/spCore.h"
#include "ip/ipCore.h"

using namespace boost::python;

void bind_ip_ipcore()
{
  class_<Torch::ipCore, bases<Torch::spCore>, boost::noncopyable>("ipCore", "The base type for all image processing operators", no_init);
}
