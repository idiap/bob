/**
 * @file python/math/src/interiorpointLP.cc
 * @date Sat Mar 19 19:49:51 2011 +0100
 * @author Laurent El Shafey <Laurent.El-Shafey@idiap.ch>
 *
 * @brief Binds the interior point methods which allow to solve a
 *        Linear Programming problem (LP).
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

#include "math/interiorpointLP.h"

#include "core/python/ndarray.h"

using namespace boost::python;
namespace tp = bob::python;
namespace ca = bob::core::array;

static const char* SHORTSTEP_DOC = "Solves the Linear Programming problem using a short step interior point method, and returns the result as a 1D numpy array.";
static const char* PREDICTORCORRECTOR_DOC = "Solves the Linear Programming problem using a predictor/corrector interior point method, and returns the result as a 1D numpy array.";
static const char* LONGSTEP_DOC = "Solves the Linear Programming problem using a short step interior point method, and returns the result as a 1D numpy array.";


static object py_shortstep(tp::const_ndarray A, tp::const_ndarray b, tp::const_ndarray c,
    const double theta, tp::const_ndarray x0, const double epsilon) 
{
  const ca::typeinfo& info = x0.type();
  blitz::Array<double,1> x0_ = x0.bz<double,1>();
  tp::ndarray x(info.dtype, info.shape[0]);
  blitz::Array<double,1> x_ = x.bz<double,1>();
  x_ = x0_;
  bob::math::interiorpointShortstepLP(A.bz<double,2>(), b.bz<double,1>(), 
    c.bz<double,1>(), theta, x_, epsilon);
  tp::ndarray xf(info.dtype, info.shape[0]/2);
  blitz::Array<double,1> xf_ = xf.bz<double,1>();
  xf_ = x_(blitz::Range(0,xf_.extent(0)-1));
  return xf.self();
}

static object py_predictorcorrector(tp::const_ndarray A, tp::const_ndarray b, tp::const_ndarray c,
    const double theta_pred, const double theta_corr, tp::const_ndarray x0, const double epsilon) 
{
  const ca::typeinfo& info = x0.type();
  blitz::Array<double,1> x0_ = x0.bz<double,1>();
  tp::ndarray x(info.dtype, info.shape[0]);
  blitz::Array<double,1> x_ = x.bz<double,1>();
  x_ = x0_;
  bob::math::interiorpointPredictorCorrectorLP(A.bz<double,2>(), b.bz<double,1>(), 
    c.bz<double,1>(), theta_pred, theta_corr, x_, epsilon);
  tp::ndarray xf(info.dtype, info.shape[0]/2);
  blitz::Array<double,1> xf_ = xf.bz<double,1>();
  xf_ = x_(blitz::Range(0,xf_.extent(0)-1));
  return xf.self();
}

static object py_longstep(tp::const_ndarray A, tp::const_ndarray b, tp::const_ndarray c,
    const double gamma, const double sigma, tp::const_ndarray x0, const double epsilon) 
{
  const ca::typeinfo& info = x0.type();
  blitz::Array<double,1> x0_ = x0.bz<double,1>();
  tp::ndarray x(info.dtype, info.shape[0]);
  blitz::Array<double,1> x_ = x.bz<double,1>();
  x_ = x0_;
  bob::math::interiorpointLongstepLP(A.bz<double,2>(), b.bz<double,1>(), 
    c.bz<double,1>(), gamma, sigma, x_, epsilon);
  tp::ndarray xf(info.dtype, info.shape[0]/2);
  blitz::Array<double,1> xf_ = xf.bz<double,1>();
  xf_ = x_(blitz::Range(0,xf_.extent(0)-1));
  return xf.self();
}


void bind_math_interiorpointLP()
{
  // Interior point methods for Linear Programming
  def("interiorpointShortstepLP", &py_shortstep, (arg("A"), arg("b"), arg("c"), arg("theta"), arg("x0"), arg("epsilon")), SHORTSTEP_DOC);
  def("interiorpointPredictorCorrectorLP", &py_predictorcorrector, (arg("A"), arg("b"), arg("c"), arg("theta_pred"), arg("theta_corr"), arg("x0"), arg("epsilon")), PREDICTORCORRECTOR_DOC);
  def("interiorpointLongstepLP", &py_longstep, (arg("A"), arg("b"), arg("c"), arg("gamma"), arg("sigma"), arg("x0"), arg("epsilon")), LONGSTEP_DOC);
}

