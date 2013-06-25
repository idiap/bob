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
  class_<bob::trainer::MLPRPropTrainer, boost::shared_ptr<bob::trainer::MLPRPropTrainer>, bases<bob::trainer::MLPBaseTrainer> >("MLPRPropTrainer", "Sets an MLP to perform discrimination based on RProp: A Direct Adaptive Method for Faster Backpropagation Learning: The RPROP Algorithm, by Martin Riedmiller and Heinrich Braun on IEEE International Conference on Neural Networks, pp. 586--591, 1993.", no_init)
    
    .def(init<const bob::trainer::MLPRPropTrainer&>((arg("self"), arg("other")), "Initializes a **new** MLPRPropTrainer copying data from another instance"))
    
    .def(init<size_t, boost::shared_ptr<bob::trainer::Cost> >((arg("self"), arg("batch_size"), arg("cost_object")),
          "Initializes a new MLPRPropTrainer trainer according to a given machine settings and a training batch size.\n" \
          "\n" \
          "Good values for batch sizes are tens of samples. RProp **is** a \"batch\" training algorithm. Do not try to set batch_size to a too-low value.\n" \
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
          "\n"
          ))
    
    .def(init<size_t, boost::shared_ptr<bob::trainer::Cost>, const bob::machine::MLP&>((arg("self"), arg("batch_size"), arg("cost_object"), arg("machine")),
          "Initializes a new MLPRPropTrainer trainer according to a given machine settings and a training batch size.\n" \
          "\n" \
          "Good values for batch sizes are tens of samples. RProp **is** a \"batch\" training algorithm. Do not try to set batch_size to a too-low value.\n" \
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
          "  An object from a derived class of :py:class:`bob.trainer.Cost` that can calculate the cost at every iteration. If you set this to ``1``, then you are implementing stochastic training.\n" \
          "\n" \
          "  .. note::\n"
          "  \n" \
          "     Good values for batch sizes are tens of samples. This may affect the convergence.\n" \
          "\n" \
          "machine\n" \
          "\n" \
          "  A :py:class:`bob.machine.MLP` object that will be used as a basis for this trainer's internal properties.\n" \
          "\n"
          ))
    
    .def(init<size_t, boost::shared_ptr<bob::trainer::Cost>, const bob::machine::MLP&, bool>((arg("self"), arg("batch_size"), arg("cost_object"), arg("machine"), arg("train_biases")),
          "Initializes a new MLPRPropTrainer trainer according to a given machine settings and a training batch size.\n" \
          "\n" \
          "Good values for batch sizes are tens of samples. RProp **is** a \"batch\" training algorithm. Do not try to set batch_size to a too-low value.\n" \
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
          "  .. note::\n" \
          "  \n" \
          "     Good values for batch sizes are tens of samples. This may affect the convergence.\n" \
          "\n" \
          "machine\n" \
          "\n" \
          "  A :py:class:`bob.machine.MLP` object that will be used as a basis for this trainer's internal properties.\n" \
          "\n" \
          "train_biases\n" \
          "\n" \
          "  A boolean indicating if we should train the biases weights (set it to ``True``) or not (set it to ``False``).\n" \
          "\n"
          ))
 
    .def(init<size_t, boost::shared_ptr<bob::trainer::Cost>, const bob::machine::MLP&>((arg("batch_size"), arg("cost"), arg("machine")), "Initializes a new MLPRPropTrainer trainer according to a given machine settings and a training batch size. Good values for batch sizes are tens of samples. RProp is a 'batch' training algorithm. Do not try to set batch_size to a too-low value."))
    
    .def(init<size_t, boost::shared_ptr<bob::trainer::Cost> >((arg("batch_size"), arg("cost")), "Creates a MLPRPropTrainer."))
    
    .def("reset", &bob::trainer::MLPRPropTrainer::reset, (arg("self")), "Re-initializes the whole training apparatus to start training a new machine. This will effectively reset all Delta matrices to their initial values and set the previous derivatives to zero as described on the section II.C of the RProp paper.")
    
    .def("train", &bob::trainer::MLPRPropTrainer::train, (arg("self"), arg("machine"), arg("input"), arg("target")), "Trains the MLP to perform discrimination using Resilient Back-propagation (R-Prop).\n" \
        "\n" \
        "Resilient Back-propagation (R-Prop) is an efficient algorithm for gradient descent with local adpatation of the weight updates, which adapts to the behaviour of the chosen error function.\n" \
        "\n" \
        "Concretely, this executes the following update rule for the weights (and biases, optionally) and respective :math:`\\Delta`'s (the current weight updates):\n" \
        "\n" \
        ".. math::\n" \
        "   \n" \
        "   \\Delta_{ij}(t) &= \\left\\{\n" \
        "     \\begin{array}{l l}\n" \
        "     \\text{min}(\\eta^+\\cdot\\Delta_{ij}(t-1), \\Delta_{\\text{max}}) & \\text{ if } \\sum_{i=1}^{N}\\frac{\\partial J(x_i; \\theta)}{\\partial \\theta_j}(t-1)\\cdot\\sum_{i=1}^{N}\\frac{\\partial J(x_i; \\theta)}{\\partial \\theta_j}(t) > 0\\\\\n" \
        "     \\max(\\eta^-\\cdot\\Delta_{ij}(t-1), \\Delta_{\\text{min}}) & \\text{ if } \\sum_{i=1}^{N}\\frac{\\partial J(x_i; \\theta)}{\\partial \\theta_j}(t-1)\\cdot\\sum_{i=1}^{N}\\frac{\\partial J(x_i; \\theta)}{\\partial \\theta_j}(t) < 0\\\\\n" \
        "     \\Delta_{ij}(t-1) & \\text{ otherwise}\n" \
        "     \\end{array}\n" \
        "   \\right. \\\\\n" \
        "   \\Delta_{ij}w(t) &= \\left\\{\n" \
        "     \\begin{array}{l l}\n" \
        "     -\\Delta_{ij}(t) & \\text{ if } \\sum_{i=1}^{N}\\frac{\\partial J(x_i; \\theta)}{\\partial \\theta_j}(t) > 0\\\\\n" \
        "     +\\Delta_{ij}(t) & \\text{ if } \\sum_{i=1}^{N}\\frac{\\partial J(x_i; \\theta)}{\\partial \\theta_j}(t) < 0\\\\\n" \
        "     0 & \\text{ otherwise}\n" \
        "     \\end{array}\n" \
        "   \\right. \\\\\n" \
        "   w_{ij}(t+1) &= w_{ij}(t) + \\Delta_{ij}(t)\n" \
        "\n" \
        "The following parameters are set *by default* and suggested by the article:\n" \
        "\n" \
        ".. math::\n" \
        "   \n" \
        "   0 < \\eta^- &< 1 < \\eta^+\\\\\n" \
        "   \\eta^- &= 0.5\\\\\n" \
        "   \\eta^+ &= 1.2\\\\\n" \
        "   \\Delta_{0} &= 0.1\\\\\n" \
        "   \\Delta_{\\text{min}} &= 10^{-6}\\\\\n" \
        "   \\Delta_{\\text{max}} &= 50.0\n" \
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
        "\n" \
        "Keyword arguments\n" \
        "\n" \
        "machine\n" \
        "\n" \
        "  The machine that will be trained. You must have called :py:meth:`bob.trainer.MLPRPropTrainer.initialize` which a similarly configured machine before being able to call this method, or an exception may be thrown.\n" \
        "\n" \
        "input\n" \
        "  A 2D :py:class:`numpy.ndarray` with 64-bit floats containing the input data for the MLP to which this training step will be based on. The matrix should be organized so each input (example) lies on a single row of ``input``.\n" \
        "\n" \
        "target\n" \
        "  A 2D :py:class:`numpy.ndarray` with 64-bit floats containing the target data for the MLP to which this training step will be based on. The matrix should be organized so each target lies on a single row of ``target``, matching each input example in ``input``.\n" \
        "\n"
        )
    
    .def("train_", &bob::trainer::MLPRPropTrainer::train_, (arg("self"), arg("machine"), arg("input"), arg("target")), "This is a version of the train() method above, which does no compatibility check on the input machine.")
    
    .add_property("deltas", &rprop_get_delta, &rprop_set_delta, "Current settings for the weight update (:math:`\\Delta_{ij}(t)`)")

    .def("set_delta", &rprop_set_delta2, (arg("self"), arg("array"), arg("k")), "Sets the delta for a given weight layer.")

    .add_property("bias_deltas", &rprop_get_delta_bias, &rprop_set_delta_bias,
        "Current settings for the bias update (:math:`\\Delta_{ij}(t)`)")

    .def("set_bias_delta", &rprop_set_delta_bias2, (arg("self"), arg("array"), arg("k")), "Sets the bias delta for a given bias layer.")
    
    .add_property("previous_derivatives", &rprop_get_prev_deriv, &rprop_set_prev_deriv, "The previous set of weight derivatives calculated by the base trainer. We keep those for the algorithm, that requires comparisons at every iteration.")

    .def("set_previous_derivative", &rprop_set_prev_deriv2, (arg("self"), arg("array"), arg("k")), "Sets the previous cost derivative for a given weight layer (index).")

    .add_property("previous_bias_derivatives", &rprop_get_prev_deriv_bias, &rprop_set_prev_deriv_bias, "The previous set of bias derivatives calculated by the base trainer. We keep those for the algorithm, that requires comparisons at every iteration.")

    .def("set_previous_bias_derivative", &rprop_set_prev_deriv_bias2, (arg("self"), arg("array"), arg("k")), "Sets the cost bias derivative for a given bias layer (index).")

    .add_property("eta_minus", &bob::trainer::MLPRPropTrainer::getEtaMinus, &bob::trainer::MLPRPropTrainer::setEtaMinus, "Learning de-enforcement parameter (defaults to 0.5)")
    .add_property("eta_plus", &bob::trainer::MLPRPropTrainer::getEtaPlus, &bob::trainer::MLPRPropTrainer::setEtaPlus, "Learning enforcement parameter (defaults to 1.2)")
    .add_property("delta_zero", &bob::trainer::MLPRPropTrainer::getDeltaZero, &bob::trainer::MLPRPropTrainer::setDeltaZero, "Initial weight update (defaults to 0.1)")
    .add_property("delta_min", &bob::trainer::MLPRPropTrainer::getDeltaMin, &bob::trainer::MLPRPropTrainer::setDeltaMin, "Minimal weight update (defaults to :math:`10^{-6}`)")
    .add_property("delta_max", &bob::trainer::MLPRPropTrainer::getDeltaMax, &bob::trainer::MLPRPropTrainer::setDeltaMax, "Maximal weight update (defaults to 50.0)")
    ;
}
