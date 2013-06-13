/**
 * @file trainer/python/cost.cc
 * @date Sat  1 Jun 10:41:23 2013 CEST
 * @author Andre Anjos <andre.anjos@idiap.ch>
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

#include <boost/python.hpp>

#include "bob/core/python/ndarray.h"
#include "bob/trainer/Cost.h"
#include "bob/trainer/SquareError.h"
#include "bob/trainer/CrossEntropyLoss.h"

using namespace boost::python;

static bool cost_is_equal(boost::shared_ptr<bob::trainer::Cost> a,
    boost::shared_ptr<bob::trainer::Cost> b) {
  return a->str() == b->str();
}

/**
 * Maps all elements of arr through function() into retval
 */
static void apply(boost::function<double (double, double)> function, bob::python::const_ndarray output, bob::python::const_ndarray target, bob::python::ndarray retval) {

  const bob::core::array::typeinfo& info = output.type();

  if (!info.is_compatible(target.type())) {
    PYTHON_ERROR(RuntimeError, "output and target arrays are incompatible - input = %s; output = %s", info.str().c_str(), target.type().str().c_str());
  }

  if (!info.is_compatible(retval.type())) {
    PYTHON_ERROR(RuntimeError, "output/target and return arrays are incompatible - input = %s; output = %s", info.str().c_str(), retval.type().str().c_str());
  }

  if (info.nd == 1) {
    blitz::Array<double,1> output_ = output.bz<double,1>();
    blitz::Array<double,1> target_ = target.bz<double,1>();
    blitz::Array<double,1> retval_ = retval.bz<double,1>();
    for (int k=0; k<output_.extent(0); ++k) 
      retval_(k) = function(output_(k), target_(k));
  }
  else if (info.nd == 2) {
    blitz::Array<double,2> output_ = output.bz<double,2>();
    blitz::Array<double,2> target_ = target.bz<double,2>();
    blitz::Array<double,2> retval_ = retval.bz<double,2>();
    for (int k=0; k<output_.extent(0); ++k) 
      for (int l=0; l<output_.extent(1); ++l)
        retval_(k,l) = function(output_(k,l), target_(k,l));
  }
  else if (info.nd == 3) {
    blitz::Array<double,3> output_ = output.bz<double,3>();
    blitz::Array<double,3> target_ = target.bz<double,3>();
    blitz::Array<double,3> retval_ = retval.bz<double,3>();
    for (int k=0; k<output_.extent(0); ++k) 
      for (int l=0; l<output_.extent(1); ++l)
        for (int m=0; m<output_.extent(2); ++m)
          retval_(k,l,m) = function(output_(k,l,m), target_(k,l,m));
  }
  else if (info.nd == 4) {
    blitz::Array<double,4> output_ = output.bz<double,4>();
    blitz::Array<double,4> target_ = target.bz<double,4>();
    blitz::Array<double,4> retval_ = retval.bz<double,4>();
    for (int k=0; k<output_.extent(0); ++k) 
      for (int l=0; l<output_.extent(1); ++l)
        for (int m=0; m<output_.extent(2); ++m)
          for (int n=0; n<output_.extent(3); ++n)
            retval_(k,l,m,n) = function(output_(k,l,m,n), target_(k,l,m,n));
  }
  else {
    PYTHON_ERROR(RuntimeError, "function only accepts 1, 2, 3 or 4-dimensional double arrays (not %dD arrays)", (int)info.nd);
  }
}

static void cost_f_ndarray_1(boost::shared_ptr<bob::trainer::Cost> c, bob::python::const_ndarray output, bob::python::const_ndarray target, bob::python::ndarray retval) {
  apply(boost::bind(&bob::trainer::Cost::f, c, _1, _2), output, target, retval);
}

static object cost_f_ndarray_2(boost::shared_ptr<bob::trainer::Cost> c, bob::python::const_ndarray output, bob::python::const_ndarray target) {
  bob::python::ndarray retval(output.type());
  cost_f_ndarray_1(c, output, target, retval);
  return retval.self();
}

static void cost_f_prime_ndarray_1(boost::shared_ptr<bob::trainer::Cost> c, bob::python::const_ndarray output, bob::python::const_ndarray target, bob::python::ndarray retval) {
  apply(boost::bind(&bob::trainer::Cost::f_prime, c, _1, _2), output, target, retval);
}

static object cost_f_prime_ndarray_2(boost::shared_ptr<bob::trainer::Cost> c, bob::python::const_ndarray output, bob::python::const_ndarray target) {
  bob::python::ndarray retval(output.type());
  cost_f_prime_ndarray_1(c, output, target, retval);
  return retval.self();
}

static void cost_error_ndarray_1(boost::shared_ptr<bob::trainer::Cost> c, bob::python::const_ndarray output, bob::python::const_ndarray target, bob::python::ndarray retval) {
  apply(boost::bind(&bob::trainer::Cost::error, c, _1, _2), output, target, retval);
}

static object cost_error_ndarray_2(boost::shared_ptr<bob::trainer::Cost> c, bob::python::const_ndarray output, bob::python::const_ndarray target) {
  bob::python::ndarray retval(output.type());
  cost_error_ndarray_1(c, output, target, retval);
  return retval.self();
}

static const char COST_F_DOC[] = \
  "Computes the cost, given the current output of the linear machine or MLP\n" \
  "and the expected output.\n" \
  "\n" \
  "Keyword arguments:\n" \
  "\n" \
  "output\n" \
  "  Real output from the linear machine or MLP\n" \
  "\n" \
  "target\n" \
  "  Target output you are training to achieve\n" \
  "\n" \
  "res (optional)\n" \
  "  Where to place the result from the calculation. Only available if the input are :py:class:`numpy.ndarray`. If the input is a :py:class:`numpy.ndarray`, then the output will also be.\n" \
  "\n" \
  "Returns the cost\n";

static const char COST_F_PRIME_DOC[] = \
  "Computes the derivative of the cost w.r.t. output.\n" \
  "\n" \
  "Keyword arguments:\n" \
  "\n" \
  "output\n" \
  "  Real output from the linear machine or MLP\n" \
  "\n" \
  "target\n" \
  "  Target output you are training to achieve\n" \
  "\n" \
  "res (optional)\n" \
  "  Where to place the result from the calculation. Only available if the input are :py:class:`numpy.ndarray`.\n" \
  "\n" \
  "Returns the calculated error. If the input is a :py:class:`numpy.ndarray`, then the output will also be.\n";

static const char COST_ERROR_DOC[] = \
  "Computes the back-propagated error for a given MLP **output**\n" \
  "layer, given its activation function and outputs - i.e., the\n" \
  "error back-propagated through the last layer neuron up to the\n" \
  "synapse connecting the last hidden layer to the output layer.\n" \
  "\n" \
  "This entry point allows for optimization in the calculation of the\n" \
  "back-propagated errors in cases where there is a possibility of\n" \
  "mathematical simplification when using a certain combination of\n" \
  "cost-function and activation. For example, using a ML-cost and a\n" \
  "logistic activation function.\n" \
  "\n" \
  "Keyword arguments:\n" \
  "\n" \
  "output\n" \
  "  Real output from the linear machine or MLP\n" \
  "\n" \
  "target\n" \
  "  Target output you are training to achieve\n" \
  "\n" \
  "res (optional)\n" \
  "  Where to place the result from the calculation. Only available if the input are :py:class:`numpy.ndarray`.\n" \
  "\n" \
  "Returns the calculated error, back-propagated to before the output. If the input is a :py:class:`numpy.ndarray`, then the output will also be.\n" \
  "neuron.\n";

static const char SQUARE_ERROR_DOC[] = \
  "Calculates the Square-Error between output and target. The square error\n" \
  "is defined as follows:\n" \
  "\n" \
  ".. math::\n" \
  "   J = \\frac{(\\hat{y} - y)^2}{2}\n" \
  "\n" \
  "where :math:`\\hat{y}` is the output estimated by your machine and\n" \
  ":math:`y` is the expected output.\n" \
  "\n" \
  "Keyword arguments:\n" \
  "\n" \
  "actfun\n" \
  "  The activation function object used at the last layer\n" \
  "\n";

static const char CROSS_ENTROPY_LOSS_DOC[] = \
  "Calculates the Cross-Entropy Loss between output and target. The cross\n" \
  "entropy loss is defined as follows:\n" \
  "\n" \
  ".. math::\n" \
  "   J = - y \\cdot \\log{(\\hat{y})} - (1-y) \\log{(1-\\hat{y})}\n" \
  "\n" \
  "where :math:`\\hat{y}` is the output estimated by your machine and\n" \
  ":math:`y` is the expected output.\n";

static const char CROSS_ENTROPY_LOSS_CONSTRUCTOR_DOC[] = \
  "Keyword arguments:\n" \
  "\n" \
  "actfun\n" \
  "  The activation function object used at the last layer. If you set this to :py:class:`bob.machine.LogisticActivation`, a mathematical simplification is possible in which backprop_error() can benefit increasing the numerical stability of the training process. The simplification goes as follows:\n" \
  "\n" \
  ".. math::\n" \
  "   b = \\delta \\cdot \\varphi'(z)\n" \
  "\n" \
  "But, for the cross-entropy loss: \n" \
  "\n" \
  ".. math::\n" \
  "   \\delta = \\frac{\\hat{y} - y}{\\hat{y}(1 - \\hat{y})}\n" \
  "\n" \
  "and :math:`\\varphi'(z) = \\hat{y} - (1 - \\hat{y})`, so:\n" \
  "\n" \
  ".. math::\n" \
  "   b = \\hat{y} - y\n" \
  "\n";

void bind_trainer_cost() {
  class_<bob::trainer::Cost, boost::shared_ptr<bob::trainer::Cost>, boost::noncopyable>("Cost", "Base class for cost functions", no_init)
    .def("f", &cost_f_ndarray_1, (arg("self"), arg("output"), arg("target"), arg("res")), COST_F_DOC)
    .def("f", &cost_f_ndarray_2, (arg("self"), arg("output"), arg("target")), COST_F_DOC)
    .def("f", &bob::trainer::Cost::f, (arg("self"), arg("output"), arg("target")), COST_F_DOC)
    .def("__call__", &cost_f_ndarray_1, (arg("self"), arg("output"), arg("target"), arg("res")), COST_F_DOC)
    .def("__call__", &cost_f_ndarray_2, (arg("self"), arg("output"), arg("target")), COST_F_DOC)
    .def("__call__", &bob::trainer::Cost::f, (arg("self"), arg("output"), arg("target")), COST_F_DOC)
    .def("f_prime", &cost_f_prime_ndarray_1, (arg("self"), arg("output"), arg("target"), arg("res")), COST_F_PRIME_DOC)
    .def("f_prime", &cost_f_prime_ndarray_2, (arg("self"), arg("output"), arg("target")), COST_F_PRIME_DOC)
    .def("f_prime", &bob::trainer::Cost::f_prime, (arg("self"), arg("output"), arg("target")), COST_F_PRIME_DOC)
    .def("error", &cost_error_ndarray_1, (arg("self"), arg("output"), arg("target"), arg("res")), COST_ERROR_DOC)
    .def("error", &cost_error_ndarray_2, (arg("self"), arg("output"), arg("target")), COST_ERROR_DOC)
    .def("error", &bob::trainer::Cost::error, (arg("self"), arg("output"), arg("target")), COST_ERROR_DOC)
    .def("__str__", &bob::trainer::Cost::str)
    .def("__eq__", &cost_is_equal)
    ;

  class_<bob::trainer::SquareError, boost::shared_ptr<bob::trainer::SquareError>, bases<bob::trainer::Cost> >("SquareError", SQUARE_ERROR_DOC, init<boost::shared_ptr<bob::machine::Activation> >((arg("self"), arg("actfun")), "Builds a new SquareError object with the specified activation function."))
    ;

  class_<bob::trainer::CrossEntropyLoss, boost::shared_ptr<bob::trainer::CrossEntropyLoss>, bases<bob::trainer::Cost> >("CrossEntropyLoss", CROSS_ENTROPY_LOSS_DOC, init<boost::shared_ptr<bob::machine::Activation> >((arg("self"), arg("actfun")), CROSS_ENTROPY_LOSS_CONSTRUCTOR_DOC))
    .add_property("logistic_activation", &bob::trainer::CrossEntropyLoss::logistic_activation, "If set to True, will calculate the error using the simplification explained in the class documentation")
    ;
}
