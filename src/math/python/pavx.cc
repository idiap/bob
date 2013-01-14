/**
 * @file math/python/pavx.cc
 * @date Sat Dec 8 20:53:50 2012 +0200
 * @author Laurent El Shafey <Laurent.El-Shafey@idiap.ch>
 *
 * @brief Binds the Pool-Adjacent-Violators Algorithm
 *
 * Copyright (C) 2011-2013 Idiap Research Institute, Martigny, Switzerland
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

#include "bob/math/pavx.h"

#include "bob/core/python/ndarray.h"
#include "bob/core/cast.h"

using namespace boost::python;

static const char* C_PAVX_DOC = "Applies the Pool-Adjacent-Violators Algorithm. The input and output arrays should have the same size. This is a simplified C++ port of the isotonic regression code made available at http://www.imsv.unibe.ch/content/staff/personalhomepages/duembgen/software/isotonicregression/index_eng.html.";
static const char* C_PAVX__DOC = "Applies the Pool-Adjacent-Violators Algorithm. The input and output arrays should have the same size. Arguments are not checked! This is a simplified C++ port of the isotonic regression code made available at http://www.imsv.unibe.ch/content/staff/personalhomepages/duembgen/software/isotonicregression/index_eng.html.";
static const char* P_PAVX_DOC = "Applies the Pool-Adjacent-Violators Algorithm. The output array is allocated and returned. This is a simplified C++ port of the isotonic regression code made available at http://www.imsv.unibe.ch/content/staff/personalhomepages/duembgen/software/isotonicregression/index_eng.html.";
static const char* P_PAVXWIDTH_DOC = "Applies the Pool-Adjacent-Violators Algorithm. The input and output arrays should have the same size. The width array is returned. This is a simplified C++ port of the isotonic regression code made available at http://www.imsv.unibe.ch/content/staff/personalhomepages/duembgen/software/isotonicregression/index_eng.html.";
static const char* P_PAVXWIDTHHEIGHT_DOC = "Applies the Pool-Adjacent-Violators Algorithm. The input and output arrays should have the same size. The width and height arrays are returned. This is a simplified C++ port of the isotonic regression code made available at http://www.imsv.unibe.ch/content/staff/personalhomepages/duembgen/software/isotonicregression/index_eng.html.";

static void c_pavx_(bob::python::const_ndarray y, bob::python::ndarray ghat) 
{
  blitz::Array<double,1> ghat_ = ghat.bz<double,1>();
  bob::math::pavx_(y.bz<double,1>(), ghat_);
}

static void c_pavx(bob::python::const_ndarray y, bob::python::ndarray ghat) 
{
  blitz::Array<double,1> ghat_ = ghat.bz<double,1>();
  bob::math::pavx(y.bz<double,1>(), ghat_);
}

static object p_pavx(bob::python::const_ndarray y) 
{
  const bob::core::array::typeinfo& info = y.type();
  blitz::Array<double,1> y_ = y.bz<double,1>();
  bob::python::ndarray ghat(bob::core::array::t_float64, info.shape[0]);
  blitz::Array<double,1> ghat_ = ghat.bz<double,1>();
  bob::math::pavx(y.bz<double,1>(), ghat_);
  return ghat.self();
}

static object p_pavxWidth(bob::python::const_ndarray y, bob::python::ndarray ghat) 
{
  blitz::Array<double,1> ghat_ = ghat.bz<double,1>();
  blitz::Array<size_t,1> w_ = bob::math::pavxWidth(y.bz<double,1>(), ghat_);
  bob::python::ndarray width(bob::core::array::t_uint64, w_.extent(0));
  blitz::Array<uint64_t,1> width_ = width.bz<uint64_t,1>();
  width_ = bob::core::cast<uint64_t>(w_); 
  return width.self();
}

static object p_pavxWidthHeight(bob::python::const_ndarray y, bob::python::ndarray ghat) 
{
  blitz::Array<double,1> ghat_ = ghat.bz<double,1>();
  std::pair<blitz::Array<size_t,1>,blitz::Array<double,1> > pair = bob::math::pavxWidthHeight(y.bz<double,1>(), ghat_);
  bob::python::ndarray width(bob::core::array::t_uint64, pair.first.extent(0));
  blitz::Array<uint64_t,1> width_ = width.bz<uint64_t,1>();
  width_ = bob::core::cast<uint64_t>(pair.first);
  bob::python::ndarray height(bob::core::array::t_float64, pair.second.extent(0));
  blitz::Array<double,1> height_ = height.bz<double,1>();
  height_ = pair.second;
  return make_tuple(width, height);
}

void bind_math_pavx()
{
  def("pavx_", &c_pavx, (arg("input"), arg("output")), C_PAVX__DOC);
  def("pavx", &c_pavx_, (arg("input"), arg("output")), C_PAVX_DOC);
  def("pavx", &p_pavx, (arg("input")), P_PAVX_DOC);
  def("pavxWidth", &p_pavxWidth, (arg("input"), arg("output")), P_PAVXWIDTH_DOC);
  def("pavxWidthHeight", &p_pavxWidthHeight, (arg("input"), arg("output")), P_PAVXWIDTHHEIGHT_DOC);
}

