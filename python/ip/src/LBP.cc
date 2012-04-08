/**
 * @file python/ip/src/LBP.cc
 * @date Tue Apr 26 17:25:41 2011 +0200
 * @author Laurent El Shafey <Laurent.El-Shafey@idiap.ch>
 *
 * @brief Binds the LBP class to python
 *
 * Copyright (C) 2011-2012 Idiap Research Institute, Martigny, Switzerland
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

#include "core/python/ndarray.h"

#include <stdint.h>
#include <vector>
#include "ip/LBP.h"
#include "ip/LBP4R.h"
#include "ip/LBP8R.h"
#include "ip/LBP16R.h"
#include "ip/LBPTopOperator.h"
#include "ip/LBPHSFeatures.h"

using namespace boost::python;
namespace ip = bob::ip;
namespace tp = bob::python;
namespace ca = bob::core::array;

static const char* lbp4r_doc = "Objects of this class, after configuration, can compute Local Binary Features using 4 neighbour pixels.";
static const char* lbp8r_doc = "Objects of this class, after configuration, can compute Local Binary Features using 8 neighbour pixels.";
static const char* lbp16r_doc = "Objects of this class, after configuration, can compute Local Binary Features using 16 neighbour pixels.";


template <typename O, typename T> 
static void inner_call_inout (O& op, tp::const_ndarray input,
    tp::ndarray output) {
  blitz::Array<uint16_t,2> out_ = output.bz<uint16_t,2>();
  op(input.bz<T,2>(), out_);
}

template <typename O>
static void call_inout (O& op, tp::const_ndarray input,
    tp::ndarray output) {
  switch(input.type().dtype) {
    case ca::t_uint8: return inner_call_inout<O,uint8_t>(op, input, output);
    case ca::t_uint16: return inner_call_inout<O,uint16_t>(op, input, output);
    case ca::t_float64: return inner_call_inout<O,double>(op, input, output);
    default:
      PYTHON_ERROR(TypeError, "LBP operator cannot process image of type '%s'", input.type().str().c_str());
  }
}

template <typename O, typename T> 
static uint16_t inner_call_pos (O& op, tp::const_ndarray input,
    int y, int x) {
  return op(input.bz<T,2>(), y, x);
}

template <typename O>
static uint16_t call_pos (O& op, tp::const_ndarray input,
    int y, int x) {
  switch(input.type().dtype) {
    case ca::t_uint8: return inner_call_pos<O,uint8_t>(op, input, y, x);
    case ca::t_uint16: return inner_call_pos<O,uint16_t>(op, input, y, x);
    case ca::t_float64: return inner_call_pos<O,double>(op, input, y, x);
    default:
      PYTHON_ERROR(TypeError, "LBP operator cannot process image of type '%s'", input.type().str().c_str());
  }
}

template <typename O, typename T> 
static object inner_call_alloc (O& op, tp::const_ndarray input) {
  blitz::Array<T,2> i_ = input.bz<T,2>();
  blitz::TinyVector<int,2> shape = op.getLBPShape(i_);
  tp::ndarray out(ca::t_uint16, shape(0), shape(1));
  blitz::Array<uint16_t,2> out_ = out.bz<uint16_t,2>();
  op(input.bz<T,2>(), out_);
  return out.self();
}

template <typename O>
static object call_alloc (O& op, tp::const_ndarray input) {
  switch(input.type().dtype) {
    case ca::t_uint8: return inner_call_alloc<O,uint8_t>(op, input);
    case ca::t_uint16: return inner_call_alloc<O,uint16_t>(op, input);
    case ca::t_float64: return inner_call_alloc<O,double>(op, input);
    default:
      PYTHON_ERROR(TypeError, "LBP operator cannot process image of type '%s'", input.type().str().c_str());
  }
}

template <typename O, typename T> 
static object inner_get_shape (O& op, tp::const_ndarray input) {
  return object(op.getLBPShape(input.bz<T,2>()));
}

template <typename O>
static object get_shape (O& op, tp::const_ndarray input) {
  switch(input.type().dtype) {
    case ca::t_uint8: return inner_get_shape<O,uint8_t>(op, input);
    case ca::t_uint16: return inner_get_shape<O,uint16_t>(op, input);
    case ca::t_float64: return inner_get_shape<O,double>(op, input);
    default:
      PYTHON_ERROR(TypeError, "LBP operator cannot process image of type '%s'", input.type().str().c_str());
  }
}

template <typename T> 
static void inner_call_lbptop (ip::LBPTopOperator& op, tp::const_ndarray input,
    tp::ndarray xy, tp::ndarray xt, tp::ndarray yt) {
  blitz::Array<uint16_t,2> xy_ = xy.bz<uint16_t,2>();
  blitz::Array<uint16_t,2> xt_ = xt.bz<uint16_t,2>();
  blitz::Array<uint16_t,2> yt_ = yt.bz<uint16_t,2>();
  op(input.bz<T,3>(), xy_, xt_, yt_);
}

static void call_lbptop (ip::LBPTopOperator& op, tp::const_ndarray input,
    tp::ndarray xy, tp::ndarray xt, tp::ndarray yt) {
  switch(input.type().dtype) {
    case ca::t_uint8: return inner_call_lbptop<uint8_t>(op, input, xy, xt, yt);
    case ca::t_uint16: return inner_call_lbptop<uint16_t>(op, input, xy, xt,yt);
    case ca::t_float64: return inner_call_lbptop<double>(op, input, xy, xt, yt);
    default:
      PYTHON_ERROR(TypeError, "LBPTop operator cannot process image of type '%s'", input.type().str().c_str());
  }
}

template <typename T> 
static object inner_lbp_apply (ip::LBPHSFeatures& op, tp::const_ndarray input) {
  std::vector<blitz::Array<uint64_t,1> > dst;
  op(input.bz<T,2>(), dst);
  list t;
  for(size_t i=0; i<dst.size(); ++i) t.append(dst[i]);
  return t;
}

static object lbp_apply (ip::LBPHSFeatures& op, tp::const_ndarray input) {
  switch(input.type().dtype) {
    case ca::t_uint8: return inner_lbp_apply<uint8_t>(op, input);
    case ca::t_uint16: return inner_lbp_apply<uint16_t>(op, input);
    case ca::t_float64: return inner_lbp_apply<double>(op, input);
    default:
      PYTHON_ERROR(TypeError, "LBPHS operator cannot process image of type '%s'", input.type().str().c_str());
  }
}

void bind_ip_lbp_new() {
  class_<ip::LBP, boost::noncopyable>("LBP", "A base class for the LBP-like operators", no_init)
    .add_property("radius", &ip::LBP::getRadius, &ip::LBP::setRadius)
    .add_property("points", &ip::LBP::getNNeighbours)
    .add_property("circular", &ip::LBP::getCircular, &ip::LBP::setCircular)
    .add_property("to_average", &ip::LBP::getToAverage, &ip::LBP::setToAverage)
    .add_property("add_average_bit", &ip::LBP::getAddAverageBit, &ip::LBP::setAddAverageBit)
    .add_property("uniform", &ip::LBP::getUniform, &ip::LBP::setUniform)
    .add_property("rotation_invariant", &ip::LBP::getRotationInvariant, &ip::LBP::setRotationInvariant)
    .add_property("elbp_type", &ip::LBP::get_eLBP, &ip::LBP::set_eLBP, "The type of extended LBP: 0 - regular LBP, 1 - transitional LBP, 2 - direction coded LBP")
    ;

  class_<ip::LBP4R, boost::shared_ptr<ip::LBP4R>, bases<ip::LBP> >("LBP4R", lbp4r_doc, init<optional<const double, const bool, const bool, const bool, const bool, const bool> >((arg("radius")=1.0,arg("circular")=false,arg("to_average")=false,arg("add_average_bit")=false,arg("uniform")=false, arg("rotation_invariant")=false, arg("elbp_type")=0), "Construct a new LBP4R object"))
    .add_property("max_label", &ip::LBP4R::getMaxLabel)
    .def("__call__", &call_inout<ip::LBP4R>, (arg("self"), arg("input"), arg("output")), "Call an object of this type to extract LBP4R features.")
    .def("__call__", &call_pos<ip::LBP4R>, (arg("self"), arg("input"), arg("y"), arg("x")), "Call an object of this type to extract LBP4R features.")
    .def("__call__", &call_alloc<ip::LBP4R>, (arg("self"), arg("input")), "Call an object of this type to extract LBP4R features.")
    .def("get_lbp_shape", &get_shape<ip::LBP4R>, (arg("self"), arg("input")), "Get a tuple containing the expected size of the output when extracting LBP4R features.")
    ;

  class_<ip::LBP8R, boost::shared_ptr<ip::LBP8R>, bases<ip::LBP> >("LBP8R", lbp8r_doc, init<optional<const double, const bool, const bool, const bool, const bool, const bool, const int> >((arg("radius")=1.0,arg("circular")=false,arg("to_average")=false,arg("add_average_bit")=false,arg("uniform")=false, arg("rotation_invariant")=false, arg("elbp_type")=0), "Construct a new LBP8R object"))
    .add_property("max_label", &ip::LBP8R::getMaxLabel)
    .def("__call__", &call_inout<ip::LBP8R>, (arg("self"), arg("input"), arg("output")), "Call an object of this type to extract LBP8R features.")
    .def("__call__", &call_pos<ip::LBP8R>, (arg("self"), arg("input"), arg("y"), arg("x")), "Call an object of this type to extract LBP8R features.")
    .def("__call__", &call_alloc<ip::LBP8R>, (arg("self"), arg("input")), "Call an object of this type to extract LBP8R features.")
    .def("get_lbp_shape", &get_shape<ip::LBP8R>, (arg("self"), arg("input")), "Get a tuple containing the expected size of the output when extracting LBP8R features.")
    ;

  class_<ip::LBP16R, boost::shared_ptr<ip::LBP16R>, bases<ip::LBP> >("LBP16R", lbp16r_doc, init<optional<const double, const bool, const bool, const bool, const bool, const bool> >((arg("radius")=1.0,arg("circular")=true,arg("to_average")=false,arg("add_average_bit")=false,arg("uniform")=false, arg("rotation_invariant")=false, arg("elbp_type")=0), "Construct a new LBP16R object"))
    .add_property("max_label", &ip::LBP16R::getMaxLabel)
    .def("__call__", &call_inout<ip::LBP16R>, (arg("self"), arg("input"), arg("output")), "Call an object of this type to extract LBP16R features.")
    .def("__call__", &call_pos<ip::LBP16R>, (arg("self"), arg("input"), arg("y"), arg("x")), "Call an object of this type to extract LBP16R features.")
    .def("__call__", &call_alloc<ip::LBP16R>, (arg("self"), arg("input")), "Call an object of this type to extract LBP16R features.")
    .def("get_lbp_shape", &get_shape<ip::LBP16R>, (arg("self"), arg("input")), "Get a tuple containing the expected size of the output when extracting LBP16R features.")
    ;

  class_<ip::LBPTopOperator, boost::shared_ptr<ip::LBPTopOperator> >("LBPTopOperator",
 "Constructs a new LBPTopOperator object starting from the algorithm configuration. Please note this object will always produce rotation invariant 2D codes, also taking into consideration pattern uniformity (u2 variant).\n\nThe radius in X (width) direction is combined with the radius in the Y (height) direction for the calculation of the LBP on the XY (frame) direction. The radius in T is taken from the number of frames input, so it is dependent on the input to LBPTopOperator::operator().\n\nThe current number of points supported in bob is either 8 or 4. Any values differing from that need implementation of specialized functionality.", init<int, int, int, int, int, int>((arg("radius_xy"), arg("points_xy"), arg("radius_xt"), arg("points_xt"),  arg("radius_yt"), arg("points_yt")), "Constructs a new ipLBPTopOperator"))
    .def("__call__", &call_lbptop, (arg("self"),arg("input"), arg("xy"), arg("xt"), arg("yt")), "Processes a 3D array representing a set of <b>grayscale</b> images and returns (by argument) the three LBP planes calculated. The 3D array has to be arranged in this way:\n\n1st dimension => frame height\n2nd dimension => frame width\n4th dimension => time\n\nThe number of frames in the array has to be always an odd number. The central frame is taken as the frame where the LBP planes have to be calculated from. The radius in dimension T (3rd dimension) is taken to be (N-1)/2 where N is the number of frames input.")
    ;

  class_<ip::LBPHSFeatures, boost::shared_ptr<ip::LBPHSFeatures> >("LBPHSFeatures", "Constructs a new LBPHSFeatures object to extract histogram of LBP over 2D blitz arrays/images.", init<const int, const int, const int, const int, optional<const double, const int, const bool, const bool, const bool, const bool, const bool> >((arg("block_h"), arg("block_w"), arg("overlap_h"), arg("overlap_w"), arg("lbp_radius")=1., arg("lbp_neighbours")=8, arg("circular")=false,arg("to_average")=false,arg("add_average_bit")=false,arg("uniform")=false, arg("rotation_invariant")=false), "Constructs a new DCT features extractor."))
    .add_property("n_bins", &ip::LBPHSFeatures::getNBins)
    .def("get_n_blocks", (const int (ip::LBPHSFeatures::*)(const blitz::Array<uint8_t,2>& src))&ip::LBPHSFeatures::getNBlocks<uint8_t>, (arg("self"),arg("input")), "Return the number of blocks generated when extracting LBPHS Features on the given input")
    .def("get_n_blocks", (const int (ip::LBPHSFeatures::*)(const blitz::Array<uint16_t,2>& src))&ip::LBPHSFeatures::getNBlocks<uint16_t>, (arg("self"),arg("input")), "Return the number of blocks generated when extracting LBPHS Features on the given input")
    .def("get_n_blocks", (const int (ip::LBPHSFeatures::*)(const blitz::Array<double,2>& src))&ip::LBPHSFeatures::getNBlocks<double>, (arg("self"),arg("input")), "Return the number of blocks generated when extracting LBPHS Features on the given input")
    .def("__call__", &lbp_apply, (arg("self"),arg("input")), "Call an object of this type to extract LBP Histogram features.")
    ;
}
