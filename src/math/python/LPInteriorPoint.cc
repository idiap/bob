/**
 * @file math/python/interiorpointLP.cc
 * @date Fri Jan 27 21:06:59 2012 +0100
 * @author Laurent El Shafey <Laurent.El-Shafey@idiap.ch>
 *
 * @brief Binds the interior point methods which allow to solve a
 *        Linear Programming problem (LP).
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

#include <bob/core/python/ndarray.h>

#include <bob/math/LPInteriorPoint.h>

using namespace boost::python;

static object get_lambda(const bob::math::LPInteriorPoint& op)
{
  bob::python::ndarray lambda(bob::core::array::t_float64, op.getDimM());
  blitz::Array<double,1> lambda_ = lambda.bz<double,1>();
  lambda_ = op.getLambda();
  return lambda.self();
}

static object get_mu(const bob::math::LPInteriorPoint& op)
{
  bob::python::ndarray mu(bob::core::array::t_float64, op.getDimN());
  blitz::Array<double,1> mu_ = mu.bz<double,1>();
  mu_ = op.getMu();
  return mu.self();
}

static object solve1(bob::math::LPInteriorPoint& op, bob::python::const_ndarray A, 
  bob::python::const_ndarray b, bob::python::const_ndarray c, 
  bob::python::const_ndarray x0)
{
  const bob::core::array::typeinfo& info = x0.type();
  if (info.dtype != bob::core::array::t_float64)
    PYTHON_ERROR(TypeError, "Linear Program solver does only support float64 type.");
  if (info.nd != 1)
    PYTHON_ERROR(TypeError, "Linear Program solver does not support more than 1 dimensions for the input solution x0.");

  // Allocate output
  blitz::Array<double,1> x(info.shape[0]);
  // Copy input solution
  x = x0.bz<double,1>();
  // Solve
  op.solve(A.bz<double,2>(), b.bz<double,1>(), c.bz<double,1>(), x);
  
  // Return result
  bob::python::ndarray res(info.dtype, info.shape[0]/2);
  blitz::Array<double,1> res_ = res.bz<double,1>();
  res_ = x(blitz::Range(0,res_.extent(0)-1));
  return res.self();
}

static object solve2(bob::math::LPInteriorPoint& op, bob::python::const_ndarray A, 
  bob::python::const_ndarray b, bob::python::const_ndarray c, 
  bob::python::const_ndarray x0, bob::python::const_ndarray lambda, 
  bob::python::const_ndarray mu)
{
  const bob::core::array::typeinfo& info = x0.type();
  if (info.dtype != bob::core::array::t_float64)
    PYTHON_ERROR(TypeError, "Linear Program solver does only support float64 type.");
  if (info.nd != 1)
    PYTHON_ERROR(TypeError, "Linear Program solver does not support more than 1 dimensions for the input solution x0.");

  // Allocate output
  blitz::Array<double,1> x(info.shape[0]);
  // Copy input solution
  x = x0.bz<double,1>();
  // Solve
  op.solve(A.bz<double,2>(), b.bz<double,1>(), c.bz<double,1>(), x, 
    lambda.bz<double,1>(), mu.bz<double,1>());

  // Return result
  bob::python::ndarray res(info.dtype, info.shape[0]/2);
  blitz::Array<double,1> res_ = res.bz<double,1>();
  res_ = x(blitz::Range(0,res_.extent(0)-1));
  return res.self();
}

static bool is_feasible(bob::math::LPInteriorPoint& op, bob::python::const_ndarray A, 
  bob::python::const_ndarray b, bob::python::const_ndarray c, 
  bob::python::const_ndarray x, bob::python::const_ndarray lambda, 
  bob::python::const_ndarray mu)
{
  return op.isFeasible(A.bz<double,2>(), b.bz<double,1>(), c.bz<double,1>(), 
    x.bz<double,1>(), lambda.bz<double,1>(), mu.bz<double,1>());
}

static bool is_in_v(bob::math::LPInteriorPoint& op, 
  bob::python::const_ndarray x, bob::python::const_ndarray mu,
  const double theta)
{
  return op.isInV(x.bz<double,1>(), mu.bz<double,1>(), theta);
}

static bool is_in_v_s(bob::math::LPInteriorPoint& op, bob::python::const_ndarray A, 
  bob::python::const_ndarray b, bob::python::const_ndarray c, 
  bob::python::const_ndarray x, bob::python::const_ndarray lambda, 
  bob::python::const_ndarray mu, const double theta)
{
  // Solve
  return op.isInVS(A.bz<double,2>(), b.bz<double,1>(), c.bz<double,1>(), 
    x.bz<double,1>(), lambda.bz<double,1>(), mu.bz<double,1>(), theta);
}

static void initialize_dual_lambda_mu(bob::math::LPInteriorPoint& op, bob::python::const_ndarray A, 
  bob::python::const_ndarray c)
{
  op.initializeDualLambdaMu(A.bz<double,2>(), c.bz<double,1>());
}

static bool is_in_vinf(bob::math::LPInteriorPointLongstep& op, 
  bob::python::const_ndarray x, bob::python::const_ndarray mu,
  const double gamma)
{
  return op.isInV(x.bz<double,1>(), mu.bz<double,1>(), gamma);
}


void bind_math_lp_interiorpoint()
{
  class_<bob::math::LPInteriorPoint, boost::shared_ptr<bob::math::LPInteriorPoint>, boost::noncopyable>("LPInteriorPoint", "A base class for the Linear Program solver based on interior point methods.\nReference:\n'Primal-Dual Interior-Point Methods',\nStephen J. Wright, ISBN: 978-0898713824, chapter 5: 'Path-Following Algorithms'", no_init)
    .def(self == self)
    .def(self != self)
    .add_property("m", &bob::math::LPInteriorPoint::getDimM, &bob::math::LPInteriorPoint::setDimM, "The first dimension M of the problem/A matrix")
    .add_property("n", &bob::math::LPInteriorPoint::getDimN, &bob::math::LPInteriorPoint::setDimN, "The second dimension N of the problem/A matrix")
    .add_property("epsilon", &bob::math::LPInteriorPoint::getEpsilon, &bob::math::LPInteriorPoint::setEpsilon, "The precision to determine whether an equality constraint is fulfilled or not")
    .add_property("lambda_", &get_lambda, "The value of the lambda dual variable (read-only)")
    .add_property("mu", &get_mu, "The value of the mu dual variable (read-only)")
    .def("reset", &bob::math::LPInteriorPoint::reset, (arg("self"), arg("M"), arg("N")), "Reset the size of the problem (M and N correspond to the dimensions of the A matrix")
    .def("solve", &solve1, (arg("self"), arg("A"), arg("b"), arg("c"), arg("x0")), "Solve a LP problem")
    .def("solve", &solve2, (arg("self"), arg("A"), arg("b"), arg("c"), arg("x0"), arg("lambda"), arg("mu")), "Solve a LP problem")
    .def("is_feasible", &is_feasible, (arg("self"), arg("A"), arg("b"), arg("c"), arg("x"), arg("lambda"), arg("mu")), "Check if a primal-dual point (x,lambda,mu) belongs to the set of feasible point (i.e. fulfill the constraints)")
    .def("is_in_v", &is_in_v, (arg("self"), arg("x"), arg("mu"), arg("theta")), "Check if a primal-dual point (x,lambda,mu) belongs to the V2 neighborhood of the central path")
    .def("is_in_v_s", &is_in_v_s, (arg("self"), arg("A"), arg("b"), arg("c"), arg("x"), arg("lambda"), arg("mu"), arg("theta")), "Check if a primal-dual point (x,lambda,mu) belongs to the V neighborhood of the central path and the set of feasible points")
    .def("initialize_dual_lambda_mu", &initialize_dual_lambda_mu, (arg("self"), arg("A"), arg("c")), "Initialize the dual variables lambda and mu by minimizing the logarithmic barrier function")
  ;

  class_<bob::math::LPInteriorPointShortstep, boost::shared_ptr<bob::math::LPInteriorPointShortstep>, bases<bob::math::LPInteriorPoint> >("LPInteriorPointShortstep", "A Linear Program solver based on a short step interior point method", init<const size_t, const size_t, const double, const double>((arg("self"), arg("M"), arg("N"), arg("theta"), arg("epsilon")), "Constructs a new LPInteriorPointShortstep solver"))
    .def(init<const bob::math::LPInteriorPointShortstep&>((arg("solver")), "Copy constructs a solver"))
    .def(self == self)
    .def(self != self)
    .add_property("theta", &bob::math::LPInteriorPointShortstep::getTheta, &bob::math::LPInteriorPointShortstep::setTheta, "The value theta used to define a V2 neighborhood")
  ;

  class_<bob::math::LPInteriorPointPredictorCorrector, boost::shared_ptr<bob::math::LPInteriorPointPredictorCorrector>, bases<bob::math::LPInteriorPoint> >("LPInteriorPointPredictorCorrector", "A Linear Program solver based on a predictor-corrector interior point method", init<const size_t, const size_t, const double, const double, const double>((arg("self"), arg("M"), arg("N"), arg("theta_pred"), arg("theta_corr"), arg("epsilon")), "Constructs a new LPInteriorPointPredictorCorrector solver"))
    .def(init<const bob::math::LPInteriorPointPredictorCorrector&>((arg("solver")), "Copy constructs a solver"))
    .def(self == self)
    .def(self != self)
    .add_property("theta_pred", &bob::math::LPInteriorPointPredictorCorrector::getThetaPred, &bob::math::LPInteriorPointPredictorCorrector::setThetaPred, "The value theta_pred used to define a V2 neighborhood")
    .add_property("theta_corr", &bob::math::LPInteriorPointPredictorCorrector::getThetaCorr, &bob::math::LPInteriorPointPredictorCorrector::setThetaCorr, "The value theta_corr used to define a V2 neighborhood")
  ;

  class_<bob::math::LPInteriorPointLongstep, boost::shared_ptr<bob::math::LPInteriorPointLongstep>, bases<bob::math::LPInteriorPoint> >("LPInteriorPointLongstep", "A Linear Program solver based on a ong step interior point method", init<const size_t, const size_t, const double, const double, const double>((arg("self"), arg("M"), arg("N"), arg("gamma"), arg("sigma"), arg("epsilon")), "Constructs a new LPInteriorPointLongstep solver"))
    .def(init<const bob::math::LPInteriorPointLongstep&>((arg("solver")), "Copy constructs a solver"))
    .def(self == self)
    .def(self != self)
    .def("is_in_v", &is_in_vinf, (arg("self"), arg("x"), arg("mu"), arg("gamma")), "Check if a primal-dual point (x,lambda,mu) belongs to the V-inf neighborhood of the central path")
    .add_property("gamma", &bob::math::LPInteriorPointLongstep::getGamma, &bob::math::LPInteriorPointLongstep::setGamma, "The value gamma used to define a V-inf neighborhood")
    .add_property("sigma", &bob::math::LPInteriorPointLongstep::getSigma, &bob::math::LPInteriorPointLongstep::setSigma, "The value sigma used to define a V-inf neighborhood")
  ;
}

