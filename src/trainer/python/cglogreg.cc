/**
 * @file trainer/python/cglogreg.cc
 * @date Sat Sep 1 21:16:00 2012 +0200
 * @author Laurent El Shafey <laurent.el-shafey@idiap.ch>
 *
 * @brief Python bindings to Linear Logistic Regression trainer
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

#include <bob/python/ndarray.h>
#include <bob/trainer/CGLogRegTrainer.h>

using namespace boost::python;

object train1(const bob::trainer::CGLogRegTrainer& t,
  bob::python::const_ndarray data1, bob::python::const_ndarray data2)
{
  bob::machine::LinearMachine m;
  t.train(m, data1.bz<double,2>(), data2.bz<double,2>());
  return object(m);
}

void train2(const bob::trainer::CGLogRegTrainer& t, bob::machine::LinearMachine& m, 
  bob::python::const_ndarray data1, bob::python::const_ndarray data2)
{
  t.train(m, data1.bz<double,2>(), data2.bz<double,2>());
}

void bind_trainer_cglogreg() 
{
  class_<bob::trainer::CGLogRegTrainer, boost::shared_ptr<bob::trainer::CGLogRegTrainer> >("CGLogRegTrainer", "Trains a linear machine to perform Linear Logistic Regression. References:\n1. A comparison of numerical optimizers for logistic regression, T. Minka, http://research.microsoft.com/en-us/um/people/minka/papers/logreg/\n2. FoCal, http://www.dsp.sun.ac.za/~nbrummer/focal/.", init<optional<const double, const double, const size_t, const double> >((arg("self"), arg("prior")=0.5, arg("convergence_threshold")=1e-5, arg("max_iterations")=10000, arg("lambda")=0.), "Initializes a new Linear Logistic Regression trainer. The training stage will place the resulting weights (and bias) in a linear machine with a single output dimension."))
    .def(init<bob::trainer::CGLogRegTrainer&>((arg("self"), arg("other"))))
    .def(self == self)
    .def(self != self)
    .add_property("prior", &bob::trainer::CGLogRegTrainer::getPrior, &bob::trainer::CGLogRegTrainer::setPrior, "The synthetic prior (should be in range ]0.,1.[.")
    .add_property("convergence_threshold", &bob::trainer::CGLogRegTrainer::getConvergenceThreshold, &bob::trainer::CGLogRegTrainer::setConvergenceThreshold, "The convergence threshold for the conjugate gradient algorithm")
    .add_property("max_iterations", &bob::trainer::CGLogRegTrainer::getMaxIterations, &bob::trainer::CGLogRegTrainer::setMaxIterations, "The maximum number of iterations for the conjugate gradient algorithm")
    .add_property("lambda", &bob::trainer::CGLogRegTrainer::getLambda, &bob::trainer::CGLogRegTrainer::setLambda, "The regularization factor lambda")
    .def("train", &train1, (arg("self"), arg("negatives"), arg("positives")), "Trains a LinearMachine to perform the Linear Logistic Regression, using two arraysets for training, one for each of the two classes (negatives vs. positives). The trained LinearMachine is returned.")
    .def("train", &train2, (arg("self"), arg("machine"), arg("negatives"), arg("positives")), "Trains a LinearMachine to perform the Linear Logistic Regression, using two arraysets for training, one for each of the two classes (negatives vs. positives).")
    ;
}
