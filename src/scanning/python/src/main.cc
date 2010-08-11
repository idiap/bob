/**
 * @file main.cc 
 * @author <a href="mailto:andre.dos.anjos@cern.ch">Andre Anjos</a> 
 *
 * @brief Combines all modules to make up the complete bindings
 */

#include <boost/python.hpp>

using namespace boost::python;

void bind_scanning_pattern();
void bind_scanning_ipgeomnorm();
void bind_scanning_scanner();
void bind_scanning_facefinder();
void bind_scanning_gtfile();
void bind_scanning_explorer();

BOOST_PYTHON_MODULE(libpytorch_scanning) {
  scope().attr("__doc__") = "Torch classes and sub-classes for scanning images";
  bind_scanning_pattern();
  bind_scanning_ipgeomnorm();
  bind_scanning_scanner();
  bind_scanning_facefinder();
  bind_scanning_gtfile();
  bind_scanning_explorer();
}
