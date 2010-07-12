/**
 * @file main.cc 
 * @author <a href="mailto:andre.dos.anjos@cern.ch">Andre Anjos</a> 
 *
 * @brief Combines all modules to make up the complete bindings
 */

#include <boost/python.hpp>

void bind_ip_color();
void bind_ip_vision();
void bind_ip_image();
void bind_ip_video();

BOOST_PYTHON_MODULE(libpytorch_ip) {
  bind_ip_color();
  bind_ip_vision();
  bind_ip_image();
  bind_ip_video();
}
