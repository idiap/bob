/**
 * @author Andre Anjos <andre.anjos@idiap.ch> 
 * @date Fri 30 Sep 10:25:02 2011 CEST
 *
 * @brief Combines all modules to make up the complete bindings
 */

#include "core/python/pycore.h"

void bind_io_buffer_numpy();
void bind_io_exception();
void bind_io_file();
void bind_io_array();
void bind_io_arrayset();

/**
void bind_io_binfile();
void bind_io_tensorfile();
void bind_io_hdf5();
void bind_io_hdf5_extras();
void bind_io_datetime();
void bind_io_video();
**/

BOOST_PYTHON_MODULE(libpytorch_io) {

  Torch::python::setup_python("Torch classes and sub-classes for io access");

  bind_io_buffer_numpy();
  bind_io_exception();
  bind_io_file();
  bind_io_array();
  bind_io_arrayset();

  /**
  bind_io_binfile();
  bind_io_tensorfile();
  bind_io_hdf5();
  bind_io_hdf5_extras();
  bind_io_datetime();
  bind_io_video();
  **/
}
