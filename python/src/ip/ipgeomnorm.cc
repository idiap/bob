/**
 * @file src/python/ip/ipgeomnorm.cc 
 * @author <a href="mailto:andre.anjos@idiap.ch">Andre Anjos</a> 
 *
 * @brief Binds ipGeomNorm to python 
 */

#include <boost/python.hpp>

#include "ip/ipGeomNorm.h"
#include "trainer/GTFile.h"

using namespace boost::python;

static list ipgn_get_points(const Torch::ipGeomNorm& n) {
  list retval;
  const Torch::sPoint2D* points = n.getNMPoints();
  for (unsigned int i=0; i<n.getGTFile()->getNPoints(); ++i) retval.append(points[i]);
  return retval;
}

void bind_ip_ipgeomnorm()
{
  class_<Torch::ipGeomNorm, bases<Torch::ipCore> >("ipGeomNorm", init<>())
    .def("loadCfg", &Torch::ipGeomNorm::loadCfg)
    .def("setGTFile", &Torch::ipGeomNorm::setGTFile)
    .def("getGTFile", &Torch::ipGeomNorm::getGTFile, return_internal_reference<>())
    .def("getNMPoints", &ipgn_get_points)
    ;
}
