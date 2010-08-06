/**
 * @file main.cc 
 * @author <a href="mailto:andre.dos.anjos@cern.ch">Andre Anjos</a> 
 *
 * @brief Combines all modules to make up the complete bindings
 */

#include <boost/python.hpp>

using namespace boost::python;

void bind_ip_color();
void bind_ip_vision();
void bind_ip_image();
void bind_ip_video();
void bind_ip_videotensor();
void bind_ip_ipgeomnorm();
void bind_ip_lbp();

BOOST_PYTHON_MODULE(libpytorch_ip) {
  scope().attr("__doc__") = "Torch trainer classes and sub-classes";
  bind_ip_color();
  bind_ip_vision();
  bind_ip_image();
#ifdef HAVE_FFMPEG
  bind_ip_video();
  bind_ip_videotensor();
#endif
//  bind_ip_ipgeomnorm();
  bind_ip_lbp();
}
