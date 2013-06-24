/**
 * @file trainer/python/mlpbase.cc
 * @date Thu 20 Jun 2013 17:38:50 CEST
 * @author Andre Anjos <andre.anjos@idiap.ch>
 *
 * @brief Python bindings for the MLPBaseTrainer
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
#include <bob/trainer/MLPBaseTrainer.h>

using namespace boost::python;

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
  const std::vector<blitz::Array<double,2> >& v = t.getDerivatives();
  list retval;
  for (size_t k=0; k<v.size(); ++k) retval.append(v[k]); //copy
  return tuple(retval);
}

static object mlpbase_get_deriv_bias(const bob::trainer::MLPBaseTrainer& t) {
  const std::vector<blitz::Array<double,1> >& v = t.getBiasDerivatives();
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
  t.setDerivatives(vdata_ref);
}

static void mlpbase_set_deriv2(bob::trainer::MLPBaseTrainer& t, 
  bob::python::const_ndarray v, const size_t k)
{
  t.setDerivative(v.bz<double,2>(), k);
}

static void mlpbase_set_deriv_bias(bob::trainer::MLPBaseTrainer& t, 
  object data)
{
  stl_input_iterator<blitz::Array<double,1> > dbegin(data), dend;
  std::vector<blitz::Array<double,1> > vdata_ref(dbegin, dend);
  t.setBiasDerivatives(vdata_ref);
}

static void mlpbase_set_deriv_bias2(bob::trainer::MLPBaseTrainer& t, 
  bob::python::const_ndarray v, const size_t k)
{
  t.setBiasDerivative(v.bz<double,1>(), k);
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

void bind_trainer_mlpbase() {
  class_<bob::trainer::MLPBaseTrainer, boost::shared_ptr<bob::trainer::MLPBaseTrainer> >("MLPBaseTrainer", "The base python class for MLP trainers based on cost derivatives.\n\nYou should use this class when you want to create your own MLP trainers and re-use the base infrastructured provided by this class, such as the computation of partial derivatives (using the ``backward_step()`` method).", no_init)

    .def(init<const bob::trainer::MLPBaseTrainer&>((arg("self"), arg("other")), "Initializes a **new** MLPBaseTrainer copying data from another instance"))

    .def(init<size_t, boost::shared_ptr<bob::trainer::Cost> >((arg("self"), arg("batch_size"), arg("cost_object")),
            "Initializes a the MLPBaseTrainer with a batch size and a cost\n" \
            "\n" \
            "Using this constructor, you must call :py:meth:`~bob.trainer.MLPBaseTrainer.initialize`, passing your own machine later on, so to resize the internal buffers correctly. In doubt, always check machine compatibility with an initialized trainer using :py:meth:`~bob.trainer.MLPBaseTrainer.is_compatible`.\n" \
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
            "     Good values for batch sizes are tens of samples. This may affect the convergence.\n"
            ))
    .def(init<size_t, boost::shared_ptr<bob::trainer::Cost>, const bob::machine::MLP&>((arg("self"), arg("batch_size"), arg("cost_object"), arg("machine")), 
            "Initializes a the MLPBaseTrainer with a batch size and a cost\n" \
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
            "  A :py:class:`bob.machine.MLP` object that will be used as a basis for this trainer's internal properties.\n"
            ))
    .def(init<size_t, boost::shared_ptr<bob::trainer::Cost>, const bob::machine::MLP&, bool>((arg("self"), arg("batch_size"), arg("cost_object"), arg("machine"), arg("train_biases")),
            "Initializes a the MLPBaseTrainer with a batch size and a cost\n" \
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
            "  A :py:class:`bob.machine.MLP` object that will be used as a basis for this trainer's internal properties.\n" \
            "\n" \
            "train_biases\n" \
            "\n" \
            "  A boolean indicating if we should train the biases weights (set it to ``True``) or not (set it to ``False``).\n"
            ))
    .add_property("batch_size", &bob::trainer::MLPBaseTrainer::getBatchSize, &bob::trainer::MLPBaseTrainer::setBatchSize)
    .add_property("cost_object", &bob::trainer::MLPBaseTrainer::getCost, &bob::trainer::MLPBaseTrainer::setCost)
    .add_property("train_biases", &bob::trainer::MLPBaseTrainer::getTrainBiases, &bob::trainer::MLPBaseTrainer::setTrainBiases)
    .def("is_compatible", &bob::trainer::MLPBaseTrainer::isCompatible, (arg("self"), arg("machine")), "Checks if a given machine is compatible with my inner settings")
    .def("initialize", &bob::trainer::MLPBaseTrainer::initialize, (arg("self"), arg("mlp")), "Initialize the training process.")
    .def("forward_step", &mlpbase_forward_step, (arg("self"), arg("mlp"), arg("input")), "Forwards a batch of data through the MLP and updates the internal buffers.")
    .def("backward_step", &mlpbase_backward_step, (arg("self"), arg("mlp"), arg("input"), arg("target")), "Backwards a batch of data through the MLP and updates the internal buffers (errors and derivatives).")
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
    .add_property("derivatives", &mlpbase_get_deriv, &mlpbase_set_deriv)
    .def("set_derivative", &mlpbase_set_deriv2, (arg("self"), arg("array"), arg("k")), "Sets the cost derivative for a given weight layer (index).")
    .add_property("bias_derivatives", &mlpbase_get_deriv_bias, &mlpbase_set_deriv_bias)
    .def("set_bias_derivative", &mlpbase_set_deriv_bias2, (arg("self"), arg("array"), arg("k")), "Sets the cost bias derivative for a given bias layer (index).")
    .def("hidden_layers", &bob::trainer::MLPBaseTrainer::numberOfHiddenLayers,
        "The number of hidden layers on the target machine.")
  ;
}
