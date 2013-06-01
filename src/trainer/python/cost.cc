/**
 "@file trainer/python/cost.cc
 "@date Sat  1 Jun 10:41:23 2013 CEST
 "@author Andre Anjos <andre.anjos@idiap.ch>
 *
 "Copyright (C) 2011-2013 Idiap Research Institute, Martigny, Switzerland
 "
 "This program is free software: you can redistribute it and/or modify
 "it under the terms of the GNU General Public License as published by
 "the Free Software Foundation, version 3 of the License.
 "
 "This program is distributed in the hope that it will be useful,
 "but WITHOUT ANY WARRANTY; without even the implied warranty of
 "MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 "GNU General Public License for more details.
 "
 "You should have received a copy of the GNU General Public License
 "along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

#include <boost/python.hpp>

#include "bob/trainer/Cost.h"
#include "bob/trainer/SquareError.h"
#include "bob/trainer/CrossEntropyLoss.h"

using namespace boost::python;

static bool cost_is_equal(boost::shared_ptr<bob::trainer::Cost> a,
    boost::shared_ptr<bob::trainer::Cost> b) {
  return a->str() == b->str();
}

static const char COST_F_DOC[] = \
  "Computes cost, given the current output of the linear machine or MLP\n" \
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
  "Returns the calculated error\n";

static const char SQUARE_ERROR_DOC[] = \
  "Calculates the Square-Error between output and target. The square error\n" \
  "is defined as follows:\n" \
  "\n" \
  ".. math::\n" \
  "   J = \frac{(\\hat{y} - y)^2}{2}\n" \
  "\n" \
  "where :math:`\\hat{y}` is the output estimated by your machine and\n" \
  ":math:`y` is the expected output.\n";

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
  "logistic_activation\n" \
  "  If set to ``True``, then assume this cost fucntion is being used with an output layer that is activated using the logistic function. In this case, a mathematical simplification is possible in which backprop_error() can benefit increasing the numerical stability of the training process. The simplification goes as follows:\n" \
  "\n" \
  ".. math::\n" \
  "   b = \\delta \\cdot \\varphi'(z)\n" \
  "\n" \
  "But, for the cross-entropy loss: \n" \
  "\n" \
  ".. math::\n" \
  "   \\delta = \\frac{\\hat{y} - y}{\\hat{y}(1 - \\hat{y}}\n" \
  "\n" \
  "and :math:`\\varphi'(z) = \\hat{y} - (1 - \\hat{y})`, so:\n" \
  "\n" \
  ".. math::\n" \
  "   b = \\hat{y} - y\n" \
  "\n";

void bind_trainer_cost() {
  class_<bob::trainer::Cost, boost::shared_ptr<bob::trainer::Cost>, boost::noncopyable>("Cost", "Base class for cost functions", no_init)
    .def("f", &bob::trainer::Cost::f, (arg("self"), arg("output"), arg("target")), COST_F_DOC)
    .def("__call__", &bob::trainer::Cost::f, (arg("self"), arg("output"), arg("arget")), COST_F_DOC)
    .def("f_prime", &bob::trainer::Cost::f_prime, (arg("self"), arg("output"), arg("target")), COST_F_PRIME_DOC)
    .def("__str__", &bob::machine::Activation::str)
    .def("__eq__", &cost_is_equal)
    ;

  class_<bob::trainer::SquareError, boost::shared_ptr<bob::trainer::SquareError>, bases<bob::trainer::Cost> >("SquareError", SQUARE_ERROR_DOC, init<>((arg("self"))))
    ;

  class_<bob::trainer::CrossEntropyLoss, boost::shared_ptr<bob::trainer::CrossEntropyLoss>, bases<bob::trainer::Cost> >("CrossEntropyLoss", CROSS_ENTROPY_LOSS_DOC, init<optional<bool> >((arg("self"), arg("logistic_activation")=true), CROSS_ENTROPY_LOSS_CONSTRUCTOR_DOC))
    .add_property("logistic_activation", &bob::trainer::CrossEntropyLoss::logistic_activation, "If set to True, will calculate the error using the simplification explained in the class documentation")
    ;
}
