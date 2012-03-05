/**
 * @file python/machine/src/svm.cc
 * @date Sat Dec 17 14:41:56 2011 +0100
 * @author André Anjos <andre.dos.anjos@gmail.com>
 *
 * @brief Bindings to our C++ SVM infrastructure.
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

#include <core/python/ndarray.h>
#include "machine/SVM.h"

using namespace boost::python;
namespace tp = bob::python;
namespace ca = bob::core::array;
namespace mach = bob::machine;

static tuple get_shape(const mach::SupportVector& m) {
  return make_tuple(m.inputSize(), m.outputSize());
}

static int predict_class(const mach::SupportVector& m,
    tp::const_ndarray input) {
  return m.predictClass(input.bz<double,1>());
}

static int predict_class_(const mach::SupportVector& m,
    tp::const_ndarray input) {
  return m.predictClass_(input.bz<double,1>());
}

static object predict_class_n(const mach::SupportVector& m,
    tp::const_ndarray input) {
  blitz::Array<double,2> i_ = input.bz<double,2>();
  if ((size_t)i_.extent(1) != m.inputSize()) {
    PYTHON_ERROR(RuntimeError, "Input array should have %lu columns, but you have given me one with %d instead", m.inputSize(), i_.extent(1));
  }
  blitz::Range all = blitz::Range::all();
  list retval;
  for (int k=0; k<i_.extent(0); ++k) {
    blitz::Array<double,1> tmp = i_(k,all);
    retval.append(m.predictClass_(tmp));
  }
  return tuple(retval);
}

static int predict_class_and_scores(const mach::SupportVector& m, 
    tp::const_ndarray input, tp::ndarray scores) {
  blitz::Array<double,1> scores_ = scores.bz<double,1>();
  return m.predictClassAndScores(input.bz<double,1>(), scores_);
}

static int predict_class_and_scores_(const mach::SupportVector& m, 
    tp::const_ndarray input, tp::ndarray scores) {
  blitz::Array<double,1> scores_ = scores.bz<double,1>();
  return m.predictClassAndScores_(input.bz<double,1>(), scores_);
}

static tuple predict_class_and_scores2(const mach::SupportVector& m, 
    tp::const_ndarray input) {
  tp::ndarray scores(ca::t_float64, m.outputSize());
  blitz::Array<double,1> scores_ = scores.bz<double,1>();
  int c = m.predictClassAndScores(input.bz<double,1>(), scores_);
  return make_tuple(c, scores.self());
}

static object predict_class_and_scores_n(const mach::SupportVector& m,
    tp::const_ndarray input) {
  blitz::Array<double,2> i_ = input.bz<double,2>();
  if ((size_t)i_.extent(1) != m.inputSize()) {
    PYTHON_ERROR(RuntimeError, "Input array should have %lu columns, but you have given me one with %d instead", m.inputSize(), i_.extent(1));
  }
  blitz::Range all = blitz::Range::all();
  list classes, scores;
  for (int k=0; k<i_.extent(0); ++k) {
    blitz::Array<double,1> tmp = i_(k,all);
    tp::ndarray s(ca::t_float64, m.outputSize());
    blitz::Array<double,1> s_ = s.bz<double,1>();
    classes.append(m.predictClassAndScores_(tmp, s_));
    scores.append(s.self());
  }
  return make_tuple(tuple(classes), tuple(scores));
}

static int predict_class_and_probs(const mach::SupportVector& m, 
    tp::const_ndarray input, tp::ndarray probs) {
  blitz::Array<double,1> probs_ = probs.bz<double,1>();
  return m.predictClassAndProbabilities(input.bz<double,1>(), probs_);
}

static int predict_class_and_probs_(const mach::SupportVector& m, 
    tp::const_ndarray input, tp::ndarray probs) {
  blitz::Array<double,1> probs_ = probs.bz<double,1>();
  return m.predictClassAndProbabilities_(input.bz<double,1>(), probs_);
}

static tuple predict_class_and_probs2(const mach::SupportVector& m, 
    tp::const_ndarray input) {
  tp::ndarray probs(ca::t_float64, m.outputSize());
  blitz::Array<double,1> probs_ = probs.bz<double,1>();
  int c = m.predictClassAndProbabilities(input.bz<double,1>(), probs_);
  return make_tuple(c, probs.self());
}

static object predict_class_and_probs_n(const mach::SupportVector& m,
    tp::const_ndarray input) {
  blitz::Array<double,2> i_ = input.bz<double,2>();
  if ((size_t)i_.extent(1) != m.inputSize()) {
    PYTHON_ERROR(RuntimeError, "Input array should have %lu columns, but you have given me one with %d instead", m.inputSize(), i_.extent(1));
  }
  if (!m.supportsProbability()) {
    PYTHON_ERROR(RuntimeError, "this SVM does not support probabilities");
  }
  blitz::Range all = blitz::Range::all();
  list classes, probs;
  for (int k=0; k<i_.extent(0); ++k) {
    blitz::Array<double,1> tmp = i_(k,all);
    tp::ndarray s(ca::t_float64, m.numberOfClasses());
    blitz::Array<double,1> s_ = s.bz<double,1>();
    classes.append(m.predictClassAndProbabilities_(tmp, s_));
    probs.append(s.self());
  }
  return make_tuple(tuple(classes), tuple(probs));
}

static tuple labels(const mach::SupportVector& m) {
  list retval;
  for (size_t k=0; k<m.numberOfClasses(); ++k) retval.append(m.classLabel(k));
  return tuple(retval);
}

static object svmfile_read(mach::SVMFile& f) {
  if (!f.good()) return object(); //None
  tp::ndarray values(ca::t_float64, f.shape());
  blitz::Array<double,1> values_ = values.bz<double,1>();
  int label;
  if (!f.read_(label, values_)) return object(); //None
  return make_tuple(label, values.self());
}

static tuple svmfile_read_all(mach::SVMFile& f) {
  list labels;
  list values;
  while (f.good()) {
    object data = svmfile_read(f);
    if (!TPY_ISNONE(data)) {
      labels.append(data[0]);
      values.append(data[1]);
    }
  }
  return make_tuple(tuple(labels), tuple(values));
}

static object svmfile_read2(mach::SVMFile& f, tp::ndarray values) {
  if (!f.good()) return object(); //None
  blitz::Array<double,1> values_ = values.bz<double,1>();
  int label;
  if (!f.read(label, values_)) return object(); //None
  return object(label);
}

static object svmfile_read2_(mach::SVMFile& f, tp::ndarray values) {
  if (!f.good()) return object(); //None
  blitz::Array<double,1> values_ = values.bz<double,1>();
  int label;
  if (!f.read_(label, values_)) return object(); //None
  return object(label);
}

static tuple svmfile_shape(mach::SVMFile& f) {
  return make_tuple(f.shape());
}

static void set_input_sub(mach::SupportVector& m, object o) {
  extract<int> int_check(o);
  extract<double> float_check(o);
  if (int_check.check()) { //is int
    m.setInputSubtraction(int_check());
  }
  else if (float_check.check()) { //is float
    m.setInputSubtraction(float_check());
  }
  else {
    //try hard-core extraction - throws TypeError, if not possible
    blitz::Array<double,1> val = extract<blitz::Array<double,1> >(o);
    m.setInputSubtraction(val);
  }
}

static void set_input_div(mach::SupportVector& m, object o) {
  extract<int> int_check(o);
  extract<double> float_check(o);
  if (int_check.check()) { //is int
    m.setInputDivision(int_check());
  }
  else if (float_check.check()) { //is float
    m.setInputDivision(float_check());
  }
  else {
    //try hard-core extraction - throws TypeError, if not possible
    blitz::Array<double,1> val = extract<blitz::Array<double,1> >(o);
    m.setInputDivision(val);
  }
}

void bind_machine_svm() {

  class_<mach::SVMFile, boost::shared_ptr<mach::SVMFile>, boost::noncopyable>("SVMFile", "Loads a given libsvm data file. The data file format, as defined on the library README is like this:\n\n  <label> <index1>:<value1> <index2>:<value2> ...\n  .\n  .\n  .\n\nThe labels are integer values, so are the indexes, starting from '1' (and not from zero as a C-programmer would expect. The values are floating point.\n\nZero values are suppressed - libsvm uses a sparse format.\n\nThis class is made available to you so you can input original libsvm files and convert them to another representation better supported. You cannot, from this object, save data or extend the current set.", init<const char*, size_t>((arg("filename"), arg("shape")), "Intializes an SVM file with the path to an existing file and the expected length of each input sample. Note that this cannot be guessed from the libsvm file because of its sparsity property."))
    .add_property("shape", &svmfile_shape, "The size of each sample in the file, as tuple with a single entry")
    .add_property("filename", make_function(&mach::SVMFile::filename, return_value_policy<copy_const_reference>()), "The name of the file being read")
    .def("reset", &mach::SVMFile::reset, (arg("self")), "Resets the current file so it starts reading from the begin once more")
    .def("read", &svmfile_read, (arg("self")), "Reads a single line from the file and returns a tuple containing the label and a numpy array of float64 elements. The numpy array has a shape as defined by the 'shape' property of this file. If the file has finished, returns None instead.")
    .def("read", &svmfile_read2, (arg("self"), arg("values")), "Reads a single line from the file, but in this variant you should pass the numpy array where the results of the readout will be stored. The array should have a shape that is identical to what is returned by the 'shape' property of this file or an error will be raised. Returns the label for the entry. If the file is finished, returns None instead.")
    .def("read_", &svmfile_read2_, (arg("self"), arg("values")), "Reads a single line from the file and stores the contents of the line in the input array. Does not check the array shape. Returns the label for the entry. If the file is finished, returns None instead.")
    .def("read_all", &svmfile_read_all, (arg("self")), "Reads all remaining entries in the file. Returns a tuple containing (label, values) tuples for each entry in the file, in order of appearance.")
    .def("good", &mach::SVMFile::good, (arg("self")), "Returns if the file is in a good state for readout. That means it has neither the eof, fail or bad bits set.")
    .def("fail", &mach::SVMFile::fail, (arg("self")), "Tells if the file has the fail or bad bit set")
    .def("eof", &mach::SVMFile::eof, (arg("self")), "Tells if the file has the eof bit set")
    ;

  enum_<mach::SupportVector::svm_t>("svm_type")
    .value("C_SVC", mach::SupportVector::C_SVC)
    .value("NU_SVC", mach::SupportVector::NU_SVC)
    .value("ONE_CLASS", mach::SupportVector::ONE_CLASS)
    .value("EPSILON_SVR", mach::SupportVector::EPSILON_SVR)
    .value("NU_SVR", mach::SupportVector::NU_SVR)
    ;

  enum_<mach::SupportVector::kernel_t>("svm_kernel_type")
    .value("LINEAR", mach::SupportVector::LINEAR)
    .value("POLY", mach::SupportVector::POLY)
    .value("RBF", mach::SupportVector::RBF)
    .value("SIGMOID", mach::SupportVector::SIGMOID)
    .value("PRECOMPUTED", mach::SupportVector::PRECOMPUTED)
    ;

  class_<mach::SupportVector, boost::shared_ptr<mach::SupportVector>, boost::noncopyable>("SupportVector", "This class can load and run an SVM generated by libsvm. Libsvm is a simple, easy-to-use, and efficient software for SVM classification and regression. It solves C-SVM classification, nu-SVM classification, one-class-SVM, epsilon-SVM regression, and nu-SVM regression. It also provides an automatic model selection tool for C-SVM classification. More information about libsvm can be found on its `website <http://www.csie.ntu.edu.tw/~cjlin/libsvm/>`_. In particular, this class covers most of the functionality provided by the command-line utility svm-predict.", no_init)
    .def(init<const char*>((arg("filename")), "Builds a new Support Vector Machine from a libsvm model file\n\nWhen you load using the libsvm model loader, note that the scaling parameters will be set to defaults (subtraction of 0.0 and division by 1.0). If you need scaling to be applied, set it individually using the appropriate methods bellow."))
    .def(init<bob::io::HDF5File&>((arg("config")), "Builds a new Support Vector Machine from an HDF5 file containing the configuration for this machine. Scaling parameters are also loaded from the file. Using this constructor assures a 100% state recovery from previous sessions."))
    .add_property("input_subtract", make_function(&mach::SupportVector::getInputSubraction, return_value_policy<copy_const_reference>()), &set_input_sub)
    .add_property("input_divide", make_function(&mach::SupportVector::getInputDivision, return_value_policy<copy_const_reference>()), &set_input_div)
    .add_property("shape", &get_shape, "Expected input and output sizes. Note the output size is always 1.") 
    .add_property("labels", &labels, "The labels this machine will output.")
    .add_property("svm_type", &mach::SupportVector::machineType, "The type of SVM machine contained")
    .add_property("kernel_type", &mach::SupportVector::kernelType, "The type of kernel used by the support vectors in this machine")
    .add_property("degree", &mach::SupportVector::polynomialDegree, "The polinomial degree, only valid if the kernel is polynomial")
    .add_property("gamma", &mach::SupportVector::gamma, "The gamma parameter for polynomial, RBF (gaussian) or sigmoidal kernels")
    .add_property("coef0", &mach::SupportVector::coefficient0, "The coefficient 0 for polynomial or sigmoidal kernels")
    .add_property("probability", &mach::SupportVector::supportsProbability, "true if this machine supports probability outputs")
    .def("predictClass", &predict_class, (arg("self"), arg("input")), "Returns the predicted class given a certain input. Checks the input data for size conformity. If the size is wrong, an exception is raised.")
    .def("predictClass_", &predict_class_, (arg("self"), arg("input")), "Returns the predicted class given a certain input. Does not check the input data and is, therefore, a little bit faster.")
    .def("predictClasses", &predict_class_n, (arg("self"), arg("input")), "Returns the predicted class given a certain input. Checks the input data for size conformity. If the size is wrong, an exception is raised. This variant accepts as input a 2D array with samples arranged in lines. The array can have as many lines as you want, but the number of columns should match the expected machine input size.")
    .def("predictClassAndScores", &predict_class_and_scores2, (arg("self"), arg("input")), "Returns the predicted class and output scores as a tuple, in this order. Checks the input and output arrays for size conformity. If the size is wrong, an exception is raised.")
    .def("predictClassAndScores", &predict_class_and_scores, (arg("self"), arg("input"), arg("scores")), "Returns the predicted class given a certain input. Returns the scores for each class in the second argument. Checks the input and output arrays for size conformity. If the size is wrong, an exception is raised.")
    .def("predictClassAndScores_", &predict_class_and_scores_, (arg("self"), arg("input"), arg("scores")), "Returns the predicted class given a certain input. Returns the scores for each class in the second argument. Checks the input and output arrays for size conformity. Does not check the input data and is, therefore, a little bit faster.")
    .def("predictClassesAndScores", &predict_class_and_scores_n, (arg("self"), arg("input")), "Returns the predicted class and output scores as a tuple, in this order. Checks the input array for size conformity. If the size is wrong, an exception is raised. This variant takes a single 2D double array as input. The samples should be organized row-wise.")
    .def("predictClassAndProbabilities", &predict_class_and_probs2, (arg("self"), arg("input")), "Returns the predicted class and probabilities in a tuple (on that order) given a certain input. The current machine has to support probabilities, otherwise an exception is raised. Checks the input array for size conformity. If the size is wrong, an exception is raised.")
    .def("predictClassAndProbabilities", &predict_class_and_probs, (arg("self"), arg("input"), arg("probabilities")), "Returns the predicted class given a certain input. If the model supports it, returns the probabilities for each class in the second argument, otherwise raises an exception. Checks the input and output arrays for size conformity. If the size is wrong, an exception is raised.")
    .def("predictClassAndProbabilities_", &predict_class_and_probs_, (arg("self"), arg("input"), arg("probabilities")), "Returns the predicted class given a certain input. This version will not run any checks, so you must be sure to pass the correct input to the classifier.")
    .def("predictClassesAndProbabilities", &predict_class_and_probs_n, (arg("self"), arg("input")), "Returns the predicted class and output probabilities for each possible class as a tuple, in this order. Checks the input array for size conformity. If the size is wrong, an exception is raised. This variant takes a single 2D double array as input. The samples should be organized row-wise.")
    .def("save", (void (mach::SupportVector::*)(const std::string&) const)&mach::SupportVector::save, (arg("self"), arg("filename")), "Saves the currently loaded model to an output file. Overwrites the file, if necessary")
    .def("save", (void (mach::SupportVector::*)(bob::io::HDF5File&) const)&mach::SupportVector::save, (arg("self"), arg("config")), "Saves the whole machine into a configuration file. This allows for a single instruction parameter loading, which includes both the model and the scaling parameters.")
    ;
}
