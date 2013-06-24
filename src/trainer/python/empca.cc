/**
 * @file trainer/python/empca.cc
 * @date Tue Oct 11 12:32:10 2011 +0200
 * @author Laurent El Shafey <Laurent.El-Shafey@idiap.ch>
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
#include <bob/python/ndarray.h>
#include <boost/shared_ptr.hpp>
#include <bob/trainer/EMPCATrainer.h>
#include <bob/machine/LinearMachine.h>

using namespace boost::python;

// TODO: python bindings with conversions ndarray's <-> blitz++ array's

void bind_trainer_empca() 
{
  typedef bob::trainer::EMTrainer<bob::machine::LinearMachine, blitz::Array<double,2> > EMTrainerLinearBase; 

  class_<EMTrainerLinearBase, boost::noncopyable>("EMTrainerLinear", "The base python class for all EM-based trainers.", no_init)
    .add_property("convergence_threshold", &EMTrainerLinearBase::getConvergenceThreshold, &EMTrainerLinearBase::setConvergenceThreshold, "Convergence threshold")
    .add_property("max_iterations", &EMTrainerLinearBase::getMaxIterations, &EMTrainerLinearBase::setMaxIterations, "Max iterations")
    .add_property("compute_likelihood_variable", &EMTrainerLinearBase::getComputeLikelihood, &EMTrainerLinearBase::setComputeLikelihood, "Indicates whether the log likelihood should be computed during EM or not")
    .add_property("rng", &EMTrainerLinearBase::getRng, &EMTrainerLinearBase::setRng, "The Mersenne Twister mt19937 random generator used for the initialization of subspaces/arrays before the EM loop.")
    .def("train", &EMTrainerLinearBase::train, (arg("machine"), arg("data")), "Trains a machine using data")
    .def("initialization", &EMTrainerLinearBase::initialization, (arg("machine"), arg("data")), "This method is called before the EM algorithm")
    .def("finalization", &EMTrainerLinearBase::finalization, (arg("machine"), arg("data")), "This method is called at the end of the EM algorithm")
    .def("e_step", &EMTrainerLinearBase::eStep, (arg("machine"), arg("data")),
       "Updates the hidden variable distribution (or the sufficient statistics) given the Machine parameters. ")
    .def("m_step", &EMTrainerLinearBase::mStep, (arg("machine"), arg("data")), "Updates the Machine parameters given the hidden variable distribution (or the sufficient statistics)")
    .def("compute_likelihood", &EMTrainerLinearBase::computeLikelihood, (arg("machine")), "Computes the current log likelihood given the hidden variable distribution (or the sufficient statistics)")
  ;

  class_<bob::trainer::EMPCATrainer, boost::noncopyable, bases<EMTrainerLinearBase> >("EMPCATrainer",
      "This class implements the EM algorithm for a Linear Machine (Probabilistic PCA).\n"
      "See Section 12.2 of Bishop, \"Pattern recognition and machine learning\", 2006", init<optional<double,size_t,bool> >((arg("convergence_threshold"), arg("max_iterations"), arg("compute_likelihood"))))
    .def(init<const bob::trainer::EMPCATrainer&>((arg("trainer")), "Copy constructs an EMPCATrainer"))
    .def(self == self)
    .def(self != self)
    .def("is_similar_to", &bob::trainer::EMPCATrainer::is_similar_to, (arg("self"), arg("other"), arg("r_epsilon")=1e-5, arg("a_epsilon")=1e-8), "Compares this EMPCATrainer with the 'other' one to be approximately the same.")
    .add_property("sigma2", &bob::trainer::EMPCATrainer::getSigma2, &bob::trainer::EMPCATrainer::setSigma2, "The noise sigma2 of the probabilistic model")
  ;
}
