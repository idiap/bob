/**
 * @file python/old/scanning/src/ipgeomnorm.cc
 * @date Wed Apr 6 14:49:40 2011 +0200
 * @author Andre Anjos <andre.anjos@idiap.ch>
 *
 * @brief Binds ipGeomNorm to python
 *
 * Copyright (C) 2011 Idiap Reasearch Institute, Martigny, Switzerland
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, version 3 of the License.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

#include <boost/python.hpp>
#include <boost/shared_ptr.hpp>

#include "ip/Image.h"
#include "scanning/ipGeomNorm.h"
#include "scanning/GTFile.h"

using namespace boost::python;

static list ipgn_get_points(const Torch::ipGeomNorm& n) {
  list retval;
  const Torch::sPoint2D* points = n.getNMPoints();
  for (int i=0; i<n.getGTFile()->getNPoints(); ++i) retval.append(points[i]);
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

void bind_scanning_ipgeomnorm()
{
  class_<Torch::ipGeomNorm, boost::shared_ptr<Torch::ipGeomNorm>, bases<Torch::ipCore> >("ipGeomNorm", "This class is designed to geometrically normalize a 2D/3D tensor,	using some ground truth points.	The normalized tensor has the same storage type and is of required size.", no_init)
    .def("__init__", make_constructor(&make_geomnorm))
    .def("setGTFile", &Torch::ipGeomNorm::setGTFile, (arg("self"), arg("gtfile")), "Sets the ground-truth file to be used for the normalization of the next image.")
    .def("getGTFile", &Torch::ipGeomNorm::getGTFile, return_internal_reference<>(), (arg("self")), "Returns the current ground-truth file being used")
    .def("getNMPoints", &ipgn_get_points, (arg("self")), "Returns all the normalized points")
    .def("getOutputImage", &ipgn_get_image, with_custodian_and_ward_postcall<0, 1>(), (arg("self")), "Pythonic extension to this class that allows the retrieval of the just processed image as a Torch.ip.Image object")
    ;
}
