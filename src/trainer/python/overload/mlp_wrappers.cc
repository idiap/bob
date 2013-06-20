/**
 * @file trainer/python/overload/mlp_wrappers.cc
 * @date Wed May 15 18:51:10 CEST 2013
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
#include <bob/core/python/ndarray.h>
#include <boost/python/stl_iterator.hpp>
#include <bob/trainer/MLPBaseTrainer.h>
#include <bob/trainer/MLPBackPropTrainer.h>
#include <bob/trainer/MLPRPropTrainer.h>
#include <boost/shared_ptr.hpp>

using namespace boost::python;

class MLPBaseTrainerWrapper: public bob::trainer::MLPBaseTrainer,
                             public wrapper<bob::trainer::MLPBaseTrainer> 
{
public:
  MLPBaseTrainerWrapper(size_t batch_size, boost::shared_ptr<bob::trainer::Cost> cost):
    bob::trainer::MLPBaseTrainer(batch_size, cost) {}

  MLPBaseTrainerWrapper(size_t batch_size, boost::shared_ptr<bob::trainer::Cost> cost, const bob::machine::MLP& machine):
    bob::trainer::MLPBaseTrainer(batch_size, cost, machine) {}

  virtual ~MLPBaseTrainerWrapper() {}
 
  void initialize(const bob::machine::MLP& machine) {
    if (override python_initialize = this->get_override("initialize")) 
    {
      python_initialize(machine);
    }
    else
      bob::trainer::MLPBaseTrainer::initialize(machine);
  }

  void d_initialize(const bob::machine::MLP& machine) {
    bob::trainer::MLPBaseTrainer::initialize(machine);
  }
};

class MLPBackPropTrainerWrapper: public bob::trainer::MLPBackPropTrainer,
                                 public wrapper<bob::trainer::MLPBackPropTrainer> 
{
public:
  MLPBackPropTrainerWrapper(size_t batch_size, boost::shared_ptr<bob::trainer::Cost> cost):
    bob::trainer::MLPBackPropTrainer(batch_size, cost) {}

  MLPBackPropTrainerWrapper(size_t batch_size, boost::shared_ptr<bob::trainer::Cost> cost, const bob::machine::MLP& machine):
    bob::trainer::MLPBackPropTrainer(batch_size, cost, machine) {}

  virtual ~MLPBackPropTrainerWrapper() {}
 
  void initialize(const bob::machine::MLP& machine) {
    if (override python_initialize = this->get_override("initialize")) 
    {
      python_initialize(machine);
    }
    else
      bob::trainer::MLPBackPropTrainer::initialize(machine);
  }

  void d_initialize(const bob::machine::MLP& machine) {
    bob::trainer::MLPBackPropTrainer::initialize(machine);
  }
};

class MLPRPropTrainerWrapper: public bob::trainer::MLPRPropTrainer,
                              public wrapper<bob::trainer::MLPRPropTrainer> 
{
public:
  MLPRPropTrainerWrapper(size_t batch_size, boost::shared_ptr<bob::trainer::Cost> cost):
    bob::trainer::MLPRPropTrainer(batch_size, cost) {}

  MLPRPropTrainerWrapper(size_t batch_size, boost::shared_ptr<bob::trainer::Cost> cost, const bob::machine::MLP& machine):
    bob::trainer::MLPRPropTrainer(batch_size, cost, machine) {}

  virtual ~MLPRPropTrainerWrapper() {}
 
  void initialize(const bob::machine::MLP& machine) {
    if (override python_initialize = this->get_override("initialize")) 
    {
      python_initialize(machine);
    }
    else
      bob::trainer::MLPRPropTrainer::initialize(machine);
  }

  void d_initialize(const bob::machine::MLP& machine) {
    bob::trainer::MLPRPropTrainer::initialize(machine);
  }
};

static double mlpbase_cost1(bob::trainer::MLPBaseTrainer& t, 
  bob::python::const_ndarray target)
{
  const blitz::Array<double,2> target_ = target.bz<double,2>();
  return t.cost(target_);
}

static double mlpbase_cost2(bob::trainer::MLPBaseTrainer& t,
  const bob::machine::MLP& m, bob::python::const_ndarray input,
  bob::python::const_ndarray target)
{
  const blitz::Array<double,2> input_ = input.bz<double,2>();
  const blitz::Array<double,2> target_ = target.bz<double,2>();
  return t.cost(m, input_, target_);
}

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

static object mlpbase_get_deriv(const bob::trainer::MLPBaseTrainer& t) {
  const std::vector<blitz::Array<double,2> >& v = t.getDeriv();
  list retval;
  for (size_t k=0; k<v.size(); ++k) retval.append(v[k]); //copy
  return tuple(retval);
}

static object mlpbase_get_deriv_bias(const bob::trainer::MLPBaseTrainer& t) {
  const std::vector<blitz::Array<double,1> >& v = t.getDerivBias();
  list retval;
  for (size_t k=0; k<v.size(); ++k) retval.append(v[k]); //copy
  return tuple(retval);
}

static void mlpbase_set_error(bob::trainer::MLPBaseTrainer& t, 
  object data)
{
  stl_input_iterator<blitz::Array<double,2> > dbegin(data), dend;
  std::vector<blitz::Array<double,2> > vdata_ref(dbegin, dend);
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
  t.setOutput(vdata_ref);
}

static void mlpbase_set_output2(bob::trainer::MLPBaseTrainer& t, 
  bob::python::const_ndarray v, const size_t k)
{
  t.setOutput(v.bz<double,2>(), k);
}

static void mlpbase_set_deriv(bob::trainer::MLPBaseTrainer& t, 
  object data)
{
  stl_input_iterator<blitz::Array<double,2> > dbegin(data), dend;
  std::vector<blitz::Array<double,2> > vdata_ref(dbegin, dend);
  t.setDeriv(vdata_ref);
}

static void mlpbase_set_deriv2(bob::trainer::MLPBaseTrainer& t, 
  bob::python::const_ndarray v, const size_t k)
{
  t.setDeriv(v.bz<double,2>(), k);
}

static void mlpbase_set_deriv_bias(bob::trainer::MLPBaseTrainer& t, 
  object data)
{
  stl_input_iterator<blitz::Array<double,1> > dbegin(data), dend;
  std::vector<blitz::Array<double,1> > vdata_ref(dbegin, dend);
  t.setDerivBias(vdata_ref);
}

static void mlpbase_set_deriv_bias2(bob::trainer::MLPBaseTrainer& t, 
  bob::python::const_ndarray v, const size_t k)
{
  t.setDerivBias(v.bz<double,1>(), k);
}

static void mlpbase_forward_step(bob::trainer::MLPBaseTrainer& t, 
  const bob::machine::MLP& m, bob::python::const_ndarray input)
{
  t.forward_step(m, input.bz<double,2>());
}

static void mlpbase_backward_step(bob::trainer::MLPBaseTrainer& t, 
  const bob::machine::MLP& m, bob::python::const_ndarray input,
  bob::python::const_ndarray target)
{
  t.backward_step(m, input.bz<double,2>(), target.bz<double,2>());
}

void bind_trainer_mlp_wrappers() {

  class_<MLPBaseTrainerWrapper, boost::noncopyable >("MLPBaseTrainer", no_init)
    .def(init<size_t, boost::shared_ptr<bob::trainer::Cost>, const bob::machine::MLP&>((arg("self"), arg("batch_size"), arg("cost"), arg("mlp")), "Creates a MLPBaseTrainer."))
    .def(init<size_t, boost::shared_ptr<bob::trainer::Cost> >((arg("self"), arg("batch_size"), arg("cost")), "Creates a MLPBaseTrainer."))
    .def("initialize", &bob::trainer::MLPBaseTrainer::initialize, &MLPBaseTrainerWrapper::d_initialize, (arg("self"), arg("mlp")), "Initialize the training process.")
    .add_property("batch_size", &bob::trainer::MLPBaseTrainer::getBatchSize, &bob::trainer::MLPBaseTrainer::setBatchSize)
    .add_property("cost", &bob::trainer::MLPBaseTrainer::getCost, &bob::trainer::MLPBaseTrainer::setCost)
    .add_property("train_biases", &bob::trainer::MLPBaseTrainer::getTrainBiases, &bob::trainer::MLPBaseTrainer::setTrainBiases)
    .def("is_compatible", &bob::trainer::MLPBaseTrainer::isCompatible, (arg("self"), arg("machine")), "Checks if a given machine is compatible with my inner settings")
    .def("forward_step", &mlpbase_forward_step, (arg("self"), arg("mlp"), arg("input")), "Forwards a batch of data through the MLP and updates the internal buffers.")
    .def("backward_step", &mlpbase_backward_step, (arg("self"), arg("mlp"), arg("target")), "Backwards a batch of data through the MLP and updates the internal buffers (errors and derivatives).")
    .def("cost", &mlpbase_cost1, (arg("self"), arg("target")), 
        "Calculates the cost for a given target\n" \
        "\n" \
        "The cost for a given target is defined as the sum of individual costs for every output in the current network, averaged over all the examples.\n" \
        "\n" \
        ".. note::\n" \
        "\n" \
        "   This variant assumes you have called forward_step() before.")
    .def("cost", &mlpbase_cost2, (arg("self"), arg("machine"), arg("input"), arg("target")),
        "Calculates the cost for a given target\n" \
        "\n" \
        "The cost for a given target is defined as the sum of individual costs for every output in the current network, averaged over all the examples.\n" \
        "\n" \
        ".. note::\n" \
        "\n" \
        "   This variant will call the forward_step() before calculating the cost. After returning, you can directly call ``backward_step()`` to evaluate the derivatives w.r.t. the cost, if you wish to do so.")
    .add_property("error", &mlpbase_get_error, &mlpbase_set_error)
    .def("set_error", &mlpbase_set_error2, (arg("self"), arg("array"), arg("k")), "Sets the error for a given index.")
    .add_property("output", &mlpbase_get_output, &mlpbase_set_output)
    .def("set_output", &mlpbase_set_output2, (arg("self"), arg("array"), arg("k")), "Sets the output for a given index.")
    .add_property("deriv", &mlpbase_get_deriv, &mlpbase_set_deriv)
    .def("set_deriv", &mlpbase_set_deriv2, (arg("self"), arg("array"), arg("k")), "Sets the derivatives of the cost for a given index.")
    .add_property("deriv_bias", &mlpbase_get_deriv_bias, &mlpbase_set_deriv_bias)
    .def("set_deriv_bias", &mlpbase_set_deriv_bias2, (arg("self"), arg("array"), arg("k")), "Sets the derivatives of the cost (biases) for a given index.")
  ;

  class_<MLPBackPropTrainerWrapper, boost::noncopyable >("MLPBackPropTrainer", "Sets an MLP to perform discrimination based on vanilla error back-propagation as defined in 'Pattern Recognition and Machine Learning' by C.M. Bishop, chapter 5 or else, 'Pattern Classification' by Duda, Hart and Stork, chapter 6.", init<size_t, boost::shared_ptr<bob::trainer::Cost>, const bob::machine::MLP&>((arg("self"), arg("batch_size"), arg("cost"), arg("machine")), "Initializes a new MLPBackPropTrainer trainer according to a given machine settings and a training batch size.\n\nGood values for batch sizes are tens of samples. BackProp is not necessarily a 'batch' training algorithm, but performs in a smoother if the batch size is larger. This may also affect the convergence.\n\n You can also change default values for the learning rate and momentum. By default we train w/o any momenta.\n\nIf you want to adjust a potential learning rate decay, you can and should do it outside the scope of this trainer, in your own way."))
    .def(init<size_t, boost::shared_ptr<bob::trainer::Cost> >((arg("self"), arg("batch_size"), arg("cost")), "Creates a MLPBackPropTrainer."))
    .add_property("batch_size", &bob::trainer::MLPBaseTrainer::getBatchSize, &bob::trainer::MLPBaseTrainer::setBatchSize)
    .add_property("train_biases", &bob::trainer::MLPBaseTrainer::getTrainBiases, &bob::trainer::MLPBaseTrainer::setTrainBiases)
    .def("is_compatible", &bob::trainer::MLPBaseTrainer::isCompatible, (arg("self"), arg("machine")), "Checks if a given machine is compatible with my inner settings")
    .def("forward_step", &mlpbase_forward_step, (arg("self"), arg("mlp"), arg("input")), "Forwards a batch of data through the MLP and updates the internal buffers.")
    .def("backward_step", &mlpbase_backward_step, (arg("self"), arg("mlp"), arg("target")), "Backwards a batch of data through the MLP and updates the internal buffers (errors and derivatives).")
    .add_property("error", &mlpbase_get_error, &mlpbase_set_error)
    .def("set_error", &mlpbase_set_error2, (arg("self"), arg("array"), arg("k")), "Sets the error for a given index.")
    .add_property("output", &mlpbase_get_output, &mlpbase_set_output)
    .def("set_output", &mlpbase_set_output2, (arg("self"), arg("array"), arg("k")), "Sets the output for a given index.")
    .add_property("deriv", &mlpbase_get_deriv, &mlpbase_set_deriv)
    .def("set_deriv", &mlpbase_set_deriv2, (arg("self"), arg("array"), arg("k")), "Sets the derivatives of the cost for a given index.")
    .add_property("deriv_bias", &mlpbase_get_deriv_bias, &mlpbase_set_deriv_bias)
    .def("set_deriv_bias", &mlpbase_set_deriv_bias2, (arg("self"), arg("array"), arg("k")), "Sets the derivatives of the cost (biases) for a given index.")
    .def("initialize", &bob::trainer::MLPBackPropTrainer::initialize, &MLPBackPropTrainerWrapper::d_initialize, (arg("self"), arg("mlp")), "Initialize the training process.")
    .def("reset", &bob::trainer::MLPBackPropTrainer::reset, (arg("self")), "Re-initializes the whole training apparatus to start training a new machine. This will effectively reset all Delta matrices to their initial values and set the previous derivatives to zero.")
    .add_property("learning_rate", &bob::trainer::MLPBackPropTrainer::getLearningRate, &bob::trainer::MLPBackPropTrainer::setLearningRate)
    .add_property("momentum", &bob::trainer::MLPBackPropTrainer::getMomentum, &bob::trainer::MLPBackPropTrainer::setMomentum)
    .def("train", &bob::trainer::MLPBackPropTrainer::train, (arg("self"), arg("machine"), arg("input"), arg("target")), "Trains the MLP to perform discrimination. The training is executed outside the machine context, but uses all the current machine layout. The given machine is updated with new weights and biases at the end of the training that is performed a single time. Iterate as much as you want to refine the training.\n\nThe machine given as input is checked for compatibility with the current initialized settings. If the two are not compatible, an exception is thrown.\n\n.. note::\n   In BackProp, training is done in batches. You should set the batch size properly at class initialization or use setBatchSize().\n\n.. note::\n   The machine is not initialized randomly at each train() call. It is your task to call random() once at the machine you want to train and then call train() as many times as you think are necessary. This design allows for a training criteria to be encoded outside the scope of this trainer and to this type to focus only on applying the training when requested to.")
    .def("train_", &bob::trainer::MLPBackPropTrainer::train_, (arg("self"), arg("machine"), arg("input"), arg("target")), "This is a version of the train() method above, which does no compatibility check on the input machine.")
  ;

  class_<MLPRPropTrainerWrapper, boost::noncopyable >("MLPRPropTrainer", "Sets an MLP to perform discrimination based on RProp: A Direct Adaptive Method for Faster Backpropagation Learning: The RPROP Algorithm, by Martin Riedmiller and Heinrich Braun on IEEE International Conference on Neural Networks, pp. 586--591, 1993.", init<size_t, boost::shared_ptr<bob::trainer::Cost>, const bob::machine::MLP&>((arg("self"), arg("batch_size"), arg("cost"), arg("machine")), "Initializes a new MLPRPropTrainer trainer according to a given machine settings and a training batch size. Good values for batch sizes are tens of samples. RProp is a 'batch' training algorithm. Do not try to set batch_size to a too-low value."))
    .def(init<size_t, boost::shared_ptr<bob::trainer::Cost> >((arg("self"), arg("batch_size"), arg("cost")), "Creates a MLPRPropTrainer."))
    .add_property("batch_size", &bob::trainer::MLPBaseTrainer::getBatchSize, &bob::trainer::MLPBaseTrainer::setBatchSize)
    .add_property("train_biases", &bob::trainer::MLPBaseTrainer::getTrainBiases, &bob::trainer::MLPBaseTrainer::setTrainBiases)
    .def("is_compatible", &bob::trainer::MLPBaseTrainer::isCompatible, (arg("self"), arg("machine")), "Checks if a given machine is compatible with my inner settings")
    .def("forward_step", &mlpbase_forward_step, (arg("self"), arg("mlp"), arg("input")), "Forwards a batch of data through the MLP and updates the internal buffers.")
    .def("backward_step", &mlpbase_backward_step, (arg("self"), arg("mlp"), arg("target")), "Backwards a batch of data through the MLP and updates the internal buffers (errors and derivatives).")
    .add_property("error", &mlpbase_get_error, &mlpbase_set_error)
    .def("set_error", &mlpbase_set_error2, (arg("self"), arg("array"), arg("k")), "Sets the error for a given index.")
    .add_property("output", &mlpbase_get_output, &mlpbase_set_output)
    .def("set_output", &mlpbase_set_output2, (arg("self"), arg("array"), arg("k")), "Sets the output for a given index.")
    .add_property("deriv", &mlpbase_get_deriv, &mlpbase_set_deriv)
    .def("set_deriv", &mlpbase_set_deriv2, (arg("self"), arg("array"), arg("k")), "Sets the derivatives of the cost for a given index.")
    .add_property("deriv_bias", &mlpbase_get_deriv_bias, &mlpbase_set_deriv_bias)
    .def("set_deriv_bias", &mlpbase_set_deriv_bias2, (arg("self"), arg("array"), arg("k")), "Sets the derivatives of the cost (biases) for a given index.")
    .def("initialize", &bob::trainer::MLPRPropTrainer::initialize, &MLPRPropTrainerWrapper::d_initialize, (arg("self"), arg("mlp")), "Initialize the training process.")
    .def("reset", &bob::trainer::MLPRPropTrainer::reset, (arg("self")), "Re-initializes the whole training apparatus to start training a new machine. This will effectively reset all Delta matrices to their initial values and set the previous derivatives to zero as described on the section II.C of the RProp paper.")
    .def("train", &bob::trainer::MLPRPropTrainer::train, (arg("self"), arg("machine"), arg("input"), arg("target")), "Trains the MLP to perform discrimination. The training is executed outside the machine context, but uses all the current machine layout. The given machine is updated with new weights and biases at the end of the training that is performed a single time. Iterate as much as you want to refine the training.\n\nThe machine given as input is checked for compatibility with the current initialized settings. If the two are not compatible, an exception is thrown.\n\n.. note::\n   In RProp, training is done in batches. You should set the batch size properly at class initialization or use setBatchSize().\n\n.. note::\n   The machine is not initialized randomly at each train() call. It is your task to call random() once at the machine you want to train and then call train() as many times as you think are necessary. This design allows for a training criteria to be encoded outside the scope of this trainer and to this type to focus only on applying the training when requested to.")
    .def("train_", &bob::trainer::MLPRPropTrainer::train_, (arg("self"), arg("machine"), arg("input"), arg("target")), "This is a version of the train() method above, which does no compatibility check on the input machine.")
    ;
}
