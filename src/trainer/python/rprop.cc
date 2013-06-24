/**
 * @file trainer/python/rprop.cc
 * @date Wed Jul 13 17:54:14 2011 +0200
 * @author Andre Anjos <andre.anjos@idiap.ch>
 *
 * @brief Python bindings for RProp training
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
#include <boost/python/stl_iterator.hpp>
#include <bob/python/ndarray.h>
#include <bob/trainer/MLPRPropTrainer.h>

using namespace boost::python;

static object rprop_get_delta(const bob::trainer::MLPRPropTrainer& t) {
  const std::vector<blitz::Array<double,2> >& v = t.getDeltas();
  list retval;
  for (size_t k=0; k<v.size(); ++k) retval.append(v[k]); //copy
  return tuple(retval);
}

static object rprop_get_delta_bias(const bob::trainer::MLPRPropTrainer& t) {
  const std::vector<blitz::Array<double,1> >& v = t.getBiasDeltas();
  list retval;
  for (size_t k=0; k<v.size(); ++k) retval.append(v[k]); //copy
  return tuple(retval);
}

static void rprop_set_delta(bob::trainer::MLPRPropTrainer& t, 
  object data)
{
  stl_input_iterator<blitz::Array<double,2> > dbegin(data), dend;
  std::vector<blitz::Array<double,2> > vdata_ref(dbegin, dend);
  t.setDeltas(vdata_ref);
}

static void rprop_set_delta2(bob::trainer::MLPRPropTrainer& t, 
  bob::python::const_ndarray v, const size_t k)
{
  t.setDelta(v.bz<double,2>(), k);
}

static void rprop_set_delta_bias(bob::trainer::MLPRPropTrainer& t, 
  object data)
{
  stl_input_iterator<blitz::Array<double,1> > dbegin(data), dend;
  std::vector<blitz::Array<double,1> > vdata_ref(dbegin, dend);
  t.setBiasDeltas(vdata_ref);
}

static void rprop_set_delta_bias2(bob::trainer::MLPRPropTrainer& t, 
  bob::python::const_ndarray v, const size_t k)
{
  t.setBiasDelta(v.bz<double,1>(), k);
}

static object rprop_get_prev_deriv(const bob::trainer::MLPRPropTrainer& t) {
  const std::vector<blitz::Array<double,2> >& v = t.getPreviousDerivatives();
  list retval;
  for (size_t k=0; k<v.size(); ++k) retval.append(v[k]); //copy
  return tuple(retval);
}

static object rprop_get_prev_deriv_bias(const bob::trainer::MLPRPropTrainer& t) {
  const std::vector<blitz::Array<double,1> >& v = t.getPreviousBiasDerivatives();
  list retval;
  for (size_t k=0; k<v.size(); ++k) retval.append(v[k]); //copy
  return tuple(retval);
}

static void rprop_set_prev_deriv(bob::trainer::MLPRPropTrainer& t, 
  object data)
{
  stl_input_iterator<blitz::Array<double,2> > dbegin(data), dend;
  std::vector<blitz::Array<double,2> > vdata_ref(dbegin, dend);
  t.setPreviousDerivatives(vdata_ref);
}

static void rprop_set_prev_deriv2(bob::trainer::MLPRPropTrainer& t, 
  bob::python::const_ndarray v, const size_t k)
{
  t.setPreviousDerivative(v.bz<double,2>(), k);
}

static void rprop_set_prev_deriv_bias(bob::trainer::MLPRPropTrainer& t, 
  object data)
{
  stl_input_iterator<blitz::Array<double,1> > dbegin(data), dend;
  std::vector<blitz::Array<double,1> > vdata_ref(dbegin, dend);
  t.setPreviousBiasDerivatives(vdata_ref);
}

static void rprop_set_prev_deriv_bias2(bob::trainer::MLPRPropTrainer& t, 
  bob::python::const_ndarray v, const size_t k)
{
  t.setPreviousBiasDerivative(v.bz<double,1>(), k);
}

void bind_trainer_rprop() {
  class_<bob::trainer::MLPRPropTrainer, boost::shared_ptr<bob::trainer::MLPRPropTrainer>, bases<bob::trainer::MLPBaseTrainer> >("MLPRPropTrainer", "Sets an MLP to perform discrimination based on RProp: A Direct Adaptive Method for Faster Backpropagation Learning: The RPROP Algorithm, by Martin Riedmiller and Heinrich Braun on IEEE International Conference on Neural Networks, pp. 586--591, 1993.", init<size_t, boost::shared_ptr<bob::trainer::Cost>, const bob::machine::MLP&>((arg("batch_size"), arg("cost"), arg("machine")), "Initializes a new MLPRPropTrainer trainer according to a given machine settings and a training batch size. Good values for batch sizes are tens of samples. RProp is a 'batch' training algorithm. Do not try to set batch_size to a too-low value."))
    
    .def(init<size_t, boost::shared_ptr<bob::trainer::Cost> >((arg("batch_size"), arg("cost")), "Creates a MLPRPropTrainer."))
    
    .def("reset", &bob::trainer::MLPRPropTrainer::reset, (arg("self")), "Re-initializes the whole training apparatus to start training a new machine. This will effectively reset all Delta matrices to their initial values and set the previous derivatives to zero as described on the section II.C of the RProp paper.")
    
    .def("train", &bob::trainer::MLPRPropTrainer::train, (arg("self"), arg("machine"), arg("input"), arg("target")), "Trains the MLP to perform discrimination using Resilient Back-propagation (R-Prop).\n" \
        "\n" \
        "The training is executed outside the machine context, but uses all the current machine layout. The given machine is updated with new weights and biases at the end of the training that is performed a single time. Iterate as much as you want to refine the training.\n" \
        "\n" \
        "The machine given as input is checked for compatibility with the current initialized settings. If the two are not compatible, an exception is thrown.\n" \
        "\n" \
        ".. note::\n" \
        "\n" \
        "   In RProp, training is done in batches. You should set the batch size properly at class initialization or use setBatchSize().\n" \
        "\n" \
        ".. note::\n" \
        "\n" \
        "   The machine is not initialized randomly at each call to this method. It is your task to call :py:meth:`bob.machine.MLP.randomize` once at the machine you want to train and then call this method as many times as you think are necessary. This design allows for a training criteria to be encoded outside the scope of this trainer and to this type to focus only on applying the training when requested to.\n" \
        "\n"
        )
    
    .def("train_", &bob::trainer::MLPRPropTrainer::train_, (arg("self"), arg("machine"), arg("input"), arg("target")), "This is a version of the train() method above, which does no compatibility check on the input machine.")
    
    .add_property("deltas", &rprop_get_delta, &rprop_set_delta)
    .def("set_delta", &rprop_set_delta2, (arg("self"), arg("array"), arg("k")), "Sets the delta for a given weight layer (index).")
    .add_property("bias_deltas", &rprop_get_delta_bias, &rprop_set_delta_bias)
    .def("set_bias_delta", &rprop_set_delta_bias2, (arg("self"), arg("array"), arg("k")), "Sets the bias delta for a given bias layer (index).")
    
    .add_property("previous_derivatives", &rprop_get_prev_deriv, &rprop_set_prev_deriv)
    .def("set_previous_derivative", &rprop_set_prev_deriv2, (arg("self"), arg("array"), arg("k")), "Sets the previous cost derivative for a given weight layer (index).")
    .add_property("previous_bias_derivatives", &rprop_get_prev_deriv_bias, &rprop_set_prev_deriv_bias)
    .def("set_previous_bias_derivative", &rprop_set_prev_deriv_bias2, (arg("self"), arg("array"), arg("k")), "Sets the cost bias derivative for a given bias layer (index).")

    ;
}
