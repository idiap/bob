/**
 * @file main.cc 
 * @author <a href="mailto:andre.dos.anjos@cern.ch">Andre Anjos</a> 
 *
 * @brief Combines all modules to make up the complete bindings
 */

#include <boost/python.hpp>

void bind_scanning_pattern();
void bind_scanning_scanner();
void bind_scanning_facefinder();

BOOST_PYTHON_MODULE(libpytorch_scanning) {
  bind_scanning_pattern();
  bind_scanning_scanner();
  bind_scanning_facefinder();
}
