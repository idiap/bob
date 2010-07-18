/**
 * @file src/python/ip/ipgeomnorm.cc 
 * @author <a href="mailto:andre.anjos@idiap.ch">Andre Anjos</a> 
 *
 * @brief Binds ipGeomNorm to python 
 */

#include <boost/python.hpp>
#include <boost/shared_ptr.hpp>

#include "ip/ipGeomNorm.h"
#include "trainer/GTFile.h"
#include "ip/Image.h"

using namespace boost::python;

static list ipgn_get_points(const Torch::ipGeomNorm& n) {
  list retval;
  const Torch::sPoint2D* points = n.getNMPoints();
  for (unsigned int i=0; i<n.getGTFile()->getNPoints(); ++i) retval.append(points[i]);
  return retval;
}

static boost::shared_ptr<Torch::ipGeomNorm> make_geomnorm(const char* cfg)
{
  boost::shared_ptr<Torch::ipGeomNorm> retval(new Torch::ipGeomNorm());
  if (retval->loadCfg(cfg)) return retval;
  return boost::shared_ptr<Torch::ipGeomNorm>();
}

static boost::shared_ptr<Torch::Image> ipgn_get_image(const Torch::ipGeomNorm& gn, int index) 
{
  boost::shared_ptr<Torch::Image> retval(new Torch::Image);
  retval->setTensor(&gn.getOutput(index));
  return retval;
}

void bind_ip_ipgeomnorm()
{
  class_<Torch::ipGeomNorm, boost::shared_ptr<Torch::ipGeomNorm>, bases<Torch::ipCore> >("ipGeomNorm", "This class is designed to geometrically normalize a 2D/3D tensor,	using some ground truth points.	The normalized tensor has the same storage type and is of required size.", no_init)
    .def("__init__", make_constructor(&make_geomnorm))
    .def("setGTFile", &Torch::ipGeomNorm::setGTFile, (arg("self"), arg("gtfile")), "Sets the ground-truth file to be used for the normalization of the next image.")
    .def("getGTFile", &Torch::ipGeomNorm::getGTFile, return_internal_reference<>(), (arg("self")), "Returns the current ground-truth file being used")
    .def("getNMPoints", &ipgn_get_points, (arg("self")), "Returns all the normalized points")
    .def("getOutputImage", &ipgn_get_image, with_custodian_and_ward_postcall<0, 1>(), (arg("self")), "Pythonic extension to this class that allows the retrieval of the just processed image as a Torch.ip.Image object")
    ;
}
