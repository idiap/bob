/**
 * @file trainer/python/backprop.cc
 * @date Mon Jul 18 18:11:22 2011 +0200
 * @author Andre Anjos <andre.anjos@idiap.ch>
 *
 * @brief Python bindings for Backprop training
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
#include <bob/core/python/ndarray.h>
#include <boost/python/stl_iterator.hpp>
#include <bob/trainer/MLPBackPropTrainer.h>

using namespace boost::python;

static object mlpbase_get_error(const bob::trainer::MLPBaseTrainer& t) {
  const std::vector<blitz::Array<double,2> >& v = t.getError();
  list retval;
  for (size_t k=0; k<v.size(); ++k) retval.append(v[k]); //copy
  return tuple(retval);
}

static object mlpbase_get_output(const bob::trainer::MLPBaseTrainer& t) {
  const std::vector<blitz::Array<double,2> >& v = t.getOutput();
  list retval;
  for (size_t k=0; k<v.size(); ++k) retval.append(v[k]); //copy
  return tuple(retval);
}

static void mlpbase_set_error(bob::trainer::MLPBaseTrainer& t, 
  object data)
{
  stl_input_iterator<blitz::Array<double,2> > dbegin(data), dend;
  std::vector<blitz::Array<double,2> > vdata_ref(dbegin, dend);
  // Calls the initialization function
  t.setError(vdata_ref);
}

static void mlpbase_set_error2(bob::trainer::MLPBaseTrainer& t, 
  bob::python::const_ndarray v, const size_t k)
{
  const blitz::Array<double,2> v_ = v.bz<double,2>();
  t.setError(v_, k);
}

static void mlpbase_set_output(bob::trainer::MLPBaseTrainer& t, 
  object data)
{
  stl_input_iterator<blitz::Array<double,2> > dbegin(data), dend;
  std::vector<blitz::Array<double,2> > vdata_ref(dbegin, dend);
  // Calls the initialization function
  t.setOutput(vdata_ref);
}

static void mlpbase_set_output2(bob::trainer::MLPBaseTrainer& t, 
  bob::python::const_ndarray v, const size_t k)
{
  t.setOutput(v.bz<double,2>(), k);
}

static void mlpbase_forward_step(bob::trainer::MLPBaseTrainer& t, 
  const bob::machine::MLP& m, bob::python::const_ndarray input)
{
  t.forward_step(m, input.bz<double,2>());
}

static void mlpbase_backward_step(bob::trainer::MLPBaseTrainer& t, 
  const bob::machine::MLP& m, bob::python::const_ndarray target)
{
  t.backward_step(m, target.bz<double,2>());
}


void bind_trainer_backprop() {
  class_<bob::trainer::MLPBaseTrainer, boost::shared_ptr<bob::trainer::MLPBaseTrainer>, boost::noncopyable>("MLPBaseTrainer", "The base python class for MLP trainers.", init<const bob::machine::MLP&, size_t>((arg("machine"), arg("batch_size"))))
    .def(init<size_t>((arg("batch_size")), "Creates a base trainer"))
    .add_property("batch_size", &bob::trainer::MLPBaseTrainer::getBatchSize, &bob::trainer::MLPBaseTrainer::setBatchSize)
    .add_property("train_biases", &bob::trainer::MLPBaseTrainer::getTrainBiases, &bob::trainer::MLPBaseTrainer::setTrainBiases)
    .def("is_compatible", &bob::trainer::MLPBaseTrainer::isCompatible, (arg("self"), arg("machine")), "Checks if a given machine is compatible with my inner settings")
    .def("initialize", &bob::trainer::MLPBaseTrainer::initialize, (arg("self"), arg("mlp")), "Initialize the training process.")
    .def("forward_step", &mlpbase_forward_step, (arg("self"), arg("mlp"), arg("input")), "Forward step -- Forwards a batch of data through the MLP and updates the internal buffers.")
    .def("backward_step", &mlpbase_backward_step, (arg("self"), arg("mlp"), arg("target")), "Backward step -- Backwards a batch of data through the MLP and updates the internal buffers.")
    .add_property("error", &mlpbase_get_error, &mlpbase_set_error)
    .def("set_error", &mlpbase_set_error2, (arg("self"), arg("array"), arg("k")), "Sets the error for a given index.")
    .add_property("output", &mlpbase_get_output, &mlpbase_set_output)
    .def("set_output", &mlpbase_set_output2, (arg("self"), arg("array"), arg("k")), "Sets the output for a given index.")
  ;

  class_<bob::trainer::MLPBackPropTrainer, boost::shared_ptr<bob::trainer::MLPBackPropTrainer>, bases<bob::trainer::MLPBaseTrainer> >("MLPBackPropTrainer", "Sets an MLP to perform discrimination based on vanilla error back-propagation as defined in 'Pattern Recognition and Machine Learning' by C.M. Bishop, chapter 5 or else, 'Pattern Classification' by Duda, Hart and Stork, chapter 6.", init<const bob::machine::MLP&, size_t>((arg("machine"), arg("batch_size")), "Initializes a new MLPBackPropTrainer trainer according to a given machine settings and a training batch size.\n\nGood values for batch sizes are tens of samples. BackProp is not necessarily a 'batch' training algorithm, but performs in a smoother if the batch size is larger. This may also affect the convergence.\n\n You can also change default values for the learning rate and momentum. By default we train w/o any momenta.\n\nIf you want to adjust a potential learning rate decay, you can and should do it outside the scope of this trainer, in your own way."))
    .def(init<size_t>((arg("batch_size")), "Creates a MLPBackPropTrainer."))
    .def("reset", &bob::trainer::MLPBackPropTrainer::reset, (arg("self")), "Re-initializes the whole training apparatus to start training a new machine. This will effectively reset all Delta matrices to their initial values and set the previous derivatives to zero.")
    .add_property("learning_rate", &bob::trainer::MLPBackPropTrainer::getLearningRate, &bob::trainer::MLPBackPropTrainer::setLearningRate)
    .add_property("momentum", &bob::trainer::MLPBackPropTrainer::getMomentum, &bob::trainer::MLPBackPropTrainer::setMomentum)
    .def("train", &bob::trainer::MLPBackPropTrainer::train, (arg("self"), arg("machine"), arg("input"), arg("target")), "Trains the MLP to perform discrimination. The training is executed outside the machine context, but uses all the current machine layout. The given machine is updated with new weights and biases at the end of the training that is performed a single time. Iterate as much as you want to refine the training.\n\nThe machine given as input is checked for compatibility with the current initialized settings. If the two are not compatible, an exception is thrown.\n\n.. note::\n   In BackProp, training is done in batches. You should set the batch size properly at class initialization or use setBatchSize().\n\n.. note::\n   The machine is not initialized randomly at each train() call. It is your task to call random() once at the machine you want to train and then call train() as many times as you think are necessary. This design allows for a training criteria to be encoded outside the scope of this trainer and to this type to focus only on applying the training when requested to.")
    .def("train_", &bob::trainer::MLPBackPropTrainer::train_, (arg("self"), arg("machine"), arg("input"), arg("target")), "This is a version of the train() method above, which does no compatibility check on the input machine.")
  ;
}
