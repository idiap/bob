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
#include <bob/python/ndarray.h>
#include <boost/python/stl_iterator.hpp>
#include <bob/trainer/MLPBackPropTrainer.h>

using namespace boost::python;

void bind_trainer_backprop() {
  class_<bob::trainer::MLPBackPropTrainer, boost::shared_ptr<bob::trainer::MLPBackPropTrainer>, bases<bob::trainer::MLPBaseTrainer> >("MLPBackPropTrainer", "Sets an MLP to perform discrimination based on vanilla error back-propagation as defined in 'Pattern Recognition and Machine Learning' by C.M. Bishop, chapter 5 or else, 'Pattern Classification' by Duda, Hart and Stork, chapter 6.", no_init)
    
    .def(init<size_t, boost::shared_ptr<bob::trainer::Cost>, const bob::machine::MLP&>((arg("self"), arg("batch_size"), arg("cost_object"), arg("machine")),
          "Initializes a new MLPBackPropTrainer trainer according to a given machine settings and a training batch size.\n" \
          "\n" \
          "Good values for batch sizes are tens of samples. BackProp is not necessarily a 'batch' training algorithm, but performs in a smoother if the batch size is larger. This may also affect the convergence.\n" \
          "\n"
          "You can also change default values for the learning rate and momentum. By default, we train w/o any momentum.\n" \
          "\n" \
          "If you want to adjust a potential learning rate decay, you can and should do it outside the scope of this trainer, in your own way.\n" \
          "\n" \
          "Keyword parameters:\n" \
          "\n" \
          "batch_size\n" \
          "\n" \
          "  The size of each batch used for the forward and backward steps, so to speed-up the training\n" \
          "\n" \
          "cost_object\n" \
          "\n" \
          "  An object from a derived class of :py:class:`bob.trainer.Cost` that can calculate the cost at every iteration. If you set this to ``1``, then you are implementing stochastic training.\n"
          "\n" \
          "  .. note::\n"
          "  \n" \
          "     Good values for batch sizes are tens of samples. This may affect the convergence.\n" \
          "\n" \
          "machine\n" \
          "\n" \
          "  A :py:class:`bob.machine.MLP` object that will be used as a basis for this trainer's internal properties."
          ))
    
    .def("reset", &bob::trainer::MLPBackPropTrainer::reset, (arg("self")), "Re-initializes the whole training apparatus to start training a new machine. This will effectively reset previous derivatives to zero.")
    
    .add_property("learning_rate", &bob::trainer::MLPBackPropTrainer::getLearningRate, &bob::trainer::MLPBackPropTrainer::setLearningRate, "The learning rate to be used for the back-propagation.")
    
    .add_property("momentum", &bob::trainer::MLPBackPropTrainer::getMomentum, &bob::trainer::MLPBackPropTrainer::setMomentum, "The momentum value to be used for the back-propagation. This value allows for some *memory* on previous weight updates to be used for the next update.")

    .def("train", &bob::trainer::MLPBackPropTrainer::train, (arg("self"), arg("machine"), arg("input"), arg("target")), 
        "Trains the MLP to perform discrimination.\n" \
        "\n" \
        "The training is executed outside the machine context, but uses all the current machine layout. The given machine is updated with new weights and biases at the end of the training that is performed a single time.\n" \
        "\n" \
        "You must iterate (in Python) as much as you want to refine the training.\n" \
        "\n" \
        "The machine given as input is checked for compatibility with the current initialized settings. If the two are not compatible, an exception is thrown.\n" \
        "\n" \
        ".. note::\n" \
        "   \n" \
        "   In BackProp, training is done in batches. You should set the batch size properly at class initialization or use setBatchSize().\n" \
        "\n" \
        ".. note::\n" \
        "   \n" \
        "   The machine is **not** initialized randomly at each call to this method. It is your task to call :py:meth:`bob.machine.MLP.randomize` once at the machine you want to train and then call this method as many times as you think is necessary. This design allows for a *stopping criteria* to be encoded outside the scope of this trainer and for this method to only focus on applying the training when requested to. Stochastic training can be executed by setting the ``batch_size`` to 1.\n" \
        "\n" \
        "Keyword arguments\n" \
        "\n" \
        "machine\n" \
        "\n" \
        "  The machine that will be trained. You must have called :py:meth:`bob.trainer.MLPBackPropTrainer.initialize` which a similarly configured machine before being able to call this method, or an exception may be thrown.\n" \
        "\n" \
        "input\n" \
        "  A 2D :py:class:`numpy.ndarray` with 64-bit floats containing the input data for the MLP to which this training step will be based on. The matrix should be organized so each input (example) lies on a single row of ``input``.\n" \
        "\n" \
        "target\n" \
        "  A 2D :py:class:`numpy.ndarray` with 64-bit floats containing the target data for the MLP to which this training step will be based on. The matrix should be organized so each target lies on a single row of ``target``, matching each input example in ``input``.\n" \
        "\n"
        )
    .def("train_", &bob::trainer::MLPBackPropTrainer::train_, (arg("self"), arg("machine"), arg("input"), arg("target")), "This is a version of the train() method above, which does no compatibility check on the input machine and can be faster.")
  ;
}
