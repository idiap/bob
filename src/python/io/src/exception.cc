/**
 * @author <a href="mailto:andre.dos.anjos@gmail.com">Andre Anjos</a> 
 * @date Wed 26 Jan 14:29:39 2011 
 *
 * @brief io exceptions 
 */

#include "io/Exception.h"
#include "core/python/exception.h"

using namespace Torch::core::python;
namespace io = Torch::io;

void bind_io_exception() {
  register_exception_translator<io::IndexError>(PyExc_IndexError);
  register_exception_translator<io::IndexError>(PyExc_NameError);
  register_exception_translator<io::TypeError>(PyExc_TypeError);
  register_exception_translator<io::UnsupportedTypeError>(PyExc_TypeError);
  register_exception_translator<io::FileNotReadable>(PyExc_IOError);
  register_exception_translator<io::ImageUnsupportedType>(PyExc_TypeError);
}
