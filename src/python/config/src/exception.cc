/**
 * @author <a href="mailto:andre.dos.anjos@gmail.com">Andre Anjos</a> 
 * @date Wed 30 Mar 13:34:49 2011 
 *
 * @brief Binds some configuration exceptions
 */


#include "config/Exception.h"
#include "core/python/exception.h"

using namespace Torch::core::python;
namespace conf = Torch::config;

void bind_config_exception() {
  register_exception_translator<conf::KeyError>(PyExc_KeyError);
  register_exception_translator<conf::UnsupportedConversion>(PyExc_TypeError);
}
