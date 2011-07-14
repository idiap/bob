/**
 * @author Andre Anjos <andre.anjos@idiap.ch>
 * @date Wed 13 Jul 2011 17:08:34 CEST
 *
 * @brief Python bindings for RProp training
 */

#include <boost/python.hpp>
#include "trainer/DataShuffler.h"
#include "trainer/MLPRPropTrainer.h"

using namespace boost::python;
namespace io = Torch::io;
namespace mach = Torch::machine;
namespace train = Torch::trainer;

static tuple call_shuffler1(train::DataShuffler& s, size_t N) {
  blitz::Array<double,2> data(N, s.getDataWidth());
  blitz::Array<double,2> target(N, s.getTargetWidth());
  s(data, target);
  return make_tuple(data, target);
}

static tuple call_shuffler2(train::DataShuffler& s, boost::mt19937& rng,
    size_t N) {
  blitz::Array<double,2> data(N, s.getDataWidth());
  blitz::Array<double,2> target(N, s.getTargetWidth());
  s(rng, data, target);
  return make_tuple(data, target);
}

static tuple stdnorm(train::DataShuffler& s) {
  blitz::Array<double,1> mean(s.getDataWidth());
  blitz::Array<double,1> stddev(s.getDataWidth());
  s.getStdNorm(mean, stddev);
  return make_tuple(mean, stddev);
}

void bind_trainer_rprop() {
  class_<train::DataShuffler>("DataShuffler", "A data shuffler is capable of being populated with data from one or multiple classes and matching target values. Once setup, the shuffer can randomly select a number of vectors and accompaning targets for the different classes, filling up user containers.\n\nData shufflers are particular useful for training neural networks.", init<const std::vector<io::Arrayset>&, const std::vector<blitz::Array<double,1> >&>((arg("data"), arg("target")), "Initializes the shuffler with some data classes and corresponding targets. Note that Arraysets must have (for the time being), a shape of (1,) and an element type == double."))
    .def("stdnorm", &train::DataShuffler::getStdNorm, (arg("self"), arg("mean"), arg("stddev")), "Calculates and returns mean and standard deviation from the input data.")
    .def("stdnorm", &stdnorm, (arg("self")), "Calculates and returns mean and standard deviation from the input data.")
    .add_property("auto_stdnorm", &train::DataShuffler::getAutoStdNorm, &train::DataShuffler::setAutoStdNorm)
    .add_property("dataWidth", &train::DataShuffler::getDataWidth)
    .add_property("targetWidth", &train::DataShuffler::getTargetWidth)
    .def("__call__", &call_shuffler1, (arg("self"), arg("N")), "Populates the output matrices (data, target) by randomly selecting N arrays from the input arraysets and matching targets in the most possible fair way. The 'data' and 'target' matrices will contain N rows and the number of columns that are dependent on input arraysets and target array widths.")
    .def("__call__", &call_shuffler2, (arg("self"), arg("rng"), arg("N")), "Populates the output matrices (data, target) by randomly selecting N arrays from the input arraysets and matching targets in the most possible fair way. The 'data' and 'target' matrices will contain N rows and the number of columns that are dependent on input arraysets and target array widths. In this version you should provide your own random number generator, already initialized.")
    .def("__call__", (void (train::DataShuffler::*)(boost::mt19937&, blitz::Array<double,2>&, blitz::Array<double,2>&))&train::DataShuffler::operator(), (arg("self"), arg("data"), arg("target")), "Populates the output matrices by randomly selecting N arrays from the input arraysets and matching targets in the most possible fair way. The 'data' and 'target' matrices will contain N rows and the number of columns that are dependent on input arraysets and target arrays.\n\nWe check don't 'data' and 'target' for size compatibility and is your responsibility to do so.")
    .def("__call__", (void (train::DataShuffler::*)(blitz::Array<double,2>&, blitz::Array<double,2>&))&train::DataShuffler::operator(), (arg("self"), arg("data"), arg("target")), "This version is a shortcut to the previous declaration of operator() that actually instantiates its own random number generator and seed it a time-based variable. We guarantee two calls will lead to different results if they are at least 1 microsecond appart (procedure uses the machine clock).")
    ;

  class_<train::MLPRPropTrainer>("MLPRPropTrainer", "Sets an MLP to perform discrimination based on RProp: A Direct Adaptive Method for Faster Backpropagation Learning: The RPROP Algorithm, by Martin Riedmiller and Heinrich Braun on IEEE International Conference on Neural Networks, pp. 586--591, 1993.", init<const mach::MLP&, size_t>((arg("machine"), arg("batch_size")), "Initializes a new MLPRPropTrainer trainer according to a given machine settings and a training batch size. Good values for batch sizes are tens of samples. RProp is a 'batch' training algorithm. Do not try to set batch_size to a too-low value."))
    .def("reset", &train::MLPRPropTrainer::reset, (arg("self")), "Re-initializes the whole training apparatus to start training a new machine. This will effectively reset all Delta matrices to their initial values and set the previous derivatives to zero as described on the section II.C of the RProp paper.")
    .add_property("batchSize", &train::MLPRPropTrainer::getBatchSize, &train::MLPRPropTrainer::setBatchSize)
    .def("isCompatible", &train::MLPRPropTrainer::isCompatible, (arg("self"), arg("machine")), "Checks if a given machine is compatible with my inner settings")
    .def("train", &train::MLPRPropTrainer::train, (arg("self"), arg("machine"), arg("input"), arg("target")), "Trains the MLP to perform discrimination. The training is executed outside the machine context, but uses all the current machine layout. The given machine is updated with new weights and biases at the end of the training that is performed a single time. Iterate as much as you want to refine the training.\n\nThe machine given as input is checked for compatibility with the current initialized settings. If the two are not compatible, an exception is thrown.\n\n.. note::\n   In RProp, training is done in batches. You should set the batch size properly at class initialization or use setBatchSize().\n\n.. note::\n   The machine is not initialized randomly at each train() call. It is your task to call random() once at the machine you want to train and then call train() as many times as you think are necessary. This design allows for a training criteria to be encoded outside the scope of this trainer and to this type to focus only on applying the training when requested to.")
    .def("train_", &train::MLPRPropTrainer::train_, (arg("self"), arg("machine"), arg("input"), arg("target")), "This is a version of the train() method above, which does no compatibility check on the input machine.")
    ;
}
