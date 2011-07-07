/**
 * @author <a href="mailto:andre.dos.anjos@gmail.com">Andre Anjos</a> 
 * @date Thu  7 Jul 09:03:51 2011 
 *
 * @brief  
 */

#include <boost/python.hpp>
#include "machine/Activation.h"

using namespace boost::python;
namespace mach = Torch::machine;

void bind_machine_activation() {
  enum_<mach::Activation>("Activation")
    .value("LINEAR", mach::LINEAR)
    .value("TANH", mach::TANH)
    .value("LOG", mach::LOG)
    ;
}
