/**
 * @file python/trainer/src/backprop.cc
 * @date Mon Jul 18 18:11:22 2011 +0200
 * @author Andre Anjos <andre.anjos@idiap.ch>
 *
 * @brief Python bindings for Backprop training
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

#include <boost/python.hpp>
#include "trainer/MLPBackPropTrainer.h"

using namespace boost::python;
namespace io = bob::io;
namespace mach = bob::machine;
namespace train = bob::trainer;

void bind_trainer_backprop() {
  class_<train::MLPBackPropTrainer>("MLPBackPropTrainer", "Sets an MLP to perform discrimination based on vanilla error back-propagation as defined in 'Pattern Recognition and Machine Learning' by C.M. Bishop, chapter 5 or else, 'Pattern Classification' by Duda, Hart and Stork, chapter 6.", init<const mach::MLP&, size_t>((arg("machine"), arg("batch_size")), "Initializes a new MLPBackPropTrainer trainer according to a given machine settings and a training batch size.\n\nGood values for batch sizes are tens of samples. BackProp is not necessarily a 'batch' training algorithm, but performs in a smoother if the batch size is larger. This may also affect the convergence.\n\n You can also change default values for the learning rate and momentum. By default we train w/o any momenta.\n\nIf you want to adjust a potential learning rate decay, you can and should do it outside the scope of this trainer, in your own way."))
    .def("reset", &train::MLPBackPropTrainer::reset, (arg("self")), "Re-initializes the whole training apparatus to start training a new machine. This will effectively reset all Delta matrices to their initial values and set the previous derivatives to zero.")
    .add_property("batch_size", &train::MLPBackPropTrainer::getBatchSize, &train::MLPBackPropTrainer::setBatchSize)
    .add_property("learning_rate", &train::MLPBackPropTrainer::getLearningRate, &train::MLPBackPropTrainer::setLearningRate)
    .add_property("momentum", &train::MLPBackPropTrainer::getMomentum, &train::MLPBackPropTrainer::setMomentum)
    .add_property("train_biases", &train::MLPBackPropTrainer::getTrainBiases, &train::MLPBackPropTrainer::setTrainBiases)
    .def("is_compatible", &train::MLPBackPropTrainer::isCompatible, (arg("self"), arg("machine")), "Checks if a given machine is compatible with my inner settings")
    .def("train", &train::MLPBackPropTrainer::train, (arg("self"), arg("machine"), arg("input"), arg("target")), "Trains the MLP to perform discrimination. The training is executed outside the machine context, but uses all the current machine layout. The given machine is updated with new weights and biases at the end of the training that is performed a single time. Iterate as much as you want to refine the training.\n\nThe machine given as input is checked for compatibility with the current initialized settings. If the two are not compatible, an exception is thrown.\n\n.. note::\n   In BackProp, training is done in batches. You should set the batch size properly at class initialization or use setBatchSize().\n\n.. note::\n   The machine is not initialized randomly at each train() call. It is your task to call random() once at the machine you want to train and then call train() as many times as you think are necessary. This design allows for a training criteria to be encoded outside the scope of this trainer and to this type to focus only on applying the training when requested to.")
    .def("train_", &train::MLPBackPropTrainer::train_, (arg("self"), arg("machine"), arg("input"), arg("target")), "This is a version of the train() method above, which does no compatibility check on the input machine.")
    ;
}
