/**
 * @file machine/python/svm.cc
 * @date Sat Dec 17 14:41:56 2011 +0100
 * @author André Anjos <andre.dos.anjos@gmail.com>
 *
 * @brief Bindings to our C++ SVM infrastructure.
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

#include <bob/python/ndarray.h>
#include <bob/machine/SVM.h>

using namespace boost::python;

static tuple get_shape(const bob::machine::SupportVector& m) {
  return make_tuple(m.inputSize(), m.outputSize());
}

static object predict_class(const bob::machine::SupportVector& m,
    bob::python::const_ndarray input) {
  return object(m.predictClass(input.bz<double,1>()));
}

static object predict_class_(const bob::machine::SupportVector& m,
    bob::python::const_ndarray input) {
  return object(m.predictClass_(input.bz<double,1>()));
}

static object predict_class_n(const bob::machine::SupportVector& m,
    bob::python::const_ndarray input) {
  blitz::Array<double,2> i_ = input.bz<double,2>();
  if ((size_t)i_.extent(1) != m.inputSize()) {
    PYTHON_ERROR(RuntimeError, "Input array should have " SIZE_T_FMT " columns, but you have given me one with %d instead", m.inputSize(), i_.extent(1));
  }
  blitz::Range all = blitz::Range::all();
  list retval;
  for (int k=0; k<i_.extent(0); ++k) {
    blitz::Array<double,1> tmp = i_(k,all);
    retval.append(m.predictClass_(tmp));
  }
  return tuple(retval);
}

static object svm_call(const bob::machine::SupportVector& m,
    bob::python::const_ndarray input) {
  switch (input.type().nd) {
    case 1:
      return predict_class(m, input);
    case 2:
      return predict_class_n(m, input);
    default:
      PYTHON_ERROR(RuntimeError, "Input array should be 1D or 2D. You passed an array with " SIZE_T_FMT " dimensions instead", input.type().nd);
  }
}

static int predict_class_and_scores(const bob::machine::SupportVector& m, 
    bob::python::const_ndarray input, bob::python::ndarray scores) {
  blitz::Array<double,1> scores_ = scores.bz<double,1>();
  return m.predictClassAndScores(input.bz<double,1>(), scores_);
}

static int predict_class_and_scores_(const bob::machine::SupportVector& m, 
    bob::python::const_ndarray input, bob::python::ndarray scores) {
  blitz::Array<double,1> scores_ = scores.bz<double,1>();
  return m.predictClassAndScores_(input.bz<double,1>(), scores_);
}

static tuple predict_class_and_scores2(const bob::machine::SupportVector& m, 
    bob::python::const_ndarray input) {
  bob::python::ndarray scores(bob::core::array::t_float64, m.outputSize());
  blitz::Array<double,1> scores_ = scores.bz<double,1>();
  int c = m.predictClassAndScores(input.bz<double,1>(), scores_);
  return make_tuple(c, scores.self());
}

static object predict_class_and_scores_n(const bob::machine::SupportVector& m,
    bob::python::const_ndarray input) {
  blitz::Array<double,2> i_ = input.bz<double,2>();
  if ((size_t)i_.extent(1) != m.inputSize()) {
    PYTHON_ERROR(RuntimeError, "Input array should have " SIZE_T_FMT " columns, but you have given me one with %d instead", m.inputSize(), i_.extent(1));
  }
  blitz::Range all = blitz::Range::all();
  list classes, scores;
  for (int k=0; k<i_.extent(0); ++k) {
    blitz::Array<double,1> tmp = i_(k,all);
    bob::python::ndarray s(bob::core::array::t_float64, m.outputSize());
    blitz::Array<double,1> s_ = s.bz<double,1>();
    classes.append(m.predictClassAndScores_(tmp, s_));
    scores.append(s.self());
  }
  return make_tuple(tuple(classes), tuple(scores));
}

static int predict_class_and_probs(const bob::machine::SupportVector& m, 
    bob::python::const_ndarray input, bob::python::ndarray probs) {
  blitz::Array<double,1> probs_ = probs.bz<double,1>();
  return m.predictClassAndProbabilities(input.bz<double,1>(), probs_);
}

static int predict_class_and_probs_(const bob::machine::SupportVector& m, 
    bob::python::const_ndarray input, bob::python::ndarray probs) {
  blitz::Array<double,1> probs_ = probs.bz<double,1>();
  return m.predictClassAndProbabilities_(input.bz<double,1>(), probs_);
}

static tuple predict_class_and_probs2(const bob::machine::SupportVector& m, 
    bob::python::const_ndarray input) {
  bob::python::ndarray probs(bob::core::array::t_float64, m.outputSize());
  blitz::Array<double,1> probs_ = probs.bz<double,1>();
  int c = m.predictClassAndProbabilities(input.bz<double,1>(), probs_);
  return make_tuple(c, probs.self());
}

static object predict_class_and_probs_n(const bob::machine::SupportVector& m,
    bob::python::const_ndarray input) {
  blitz::Array<double,2> i_ = input.bz<double,2>();
  if ((size_t)i_.extent(1) != m.inputSize()) {
    PYTHON_ERROR(RuntimeError, "Input array should have " SIZE_T_FMT " columns, but you have given me one with %d instead", m.inputSize(), i_.extent(1));
  }
  if (!m.supportsProbability()) {
    PYTHON_ERROR(RuntimeError, "this SVM does not support probabilities");
  }
  blitz::Range all = blitz::Range::all();
  list classes, probs;
  for (int k=0; k<i_.extent(0); ++k) {
    blitz::Array<double,1> tmp = i_(k,all);
    bob::python::ndarray s(bob::core::array::t_float64, m.numberOfClasses());
    blitz::Array<double,1> s_ = s.bz<double,1>();
    classes.append(m.predictClassAndProbabilities_(tmp, s_));
    probs.append(s.self());
  }
  return make_tuple(tuple(classes), tuple(probs));
}

static tuple labels(const bob::machine::SupportVector& m) {
  list retval;
  for (size_t k=0; k<m.numberOfClasses(); ++k) retval.append(m.classLabel(k));
  return tuple(retval);
}

static object svmfile_read(bob::machine::SVMFile& f) {
  if (!f.good()) return object(); //None
  bob::python::ndarray values(bob::core::array::t_float64, f.shape());
  blitz::Array<double,1> values_ = values.bz<double,1>();
  int label;
  if (!f.read_(label, values_)) return object(); //None
  return make_tuple(label, values.self());
}

static tuple svmfile_read_all(bob::machine::SVMFile& f) {
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

static object svmfile_read2(bob::machine::SVMFile& f, bob::python::ndarray values) {
  if (!f.good()) return object(); //None
  blitz::Array<double,1> values_ = values.bz<double,1>();
  int label;
  if (!f.read(label, values_)) return object(); //None
  return object(label);
}

static object svmfile_read2_(bob::machine::SVMFile& f, bob::python::ndarray values) {
  if (!f.good()) return object(); //None
  blitz::Array<double,1> values_ = values.bz<double,1>();
  int label;
  if (!f.read_(label, values_)) return object(); //None
  return object(label);
}

static tuple svmfile_shape(bob::machine::SVMFile& f) {
  return make_tuple(f.shape());
}

static void set_input_sub(bob::machine::SupportVector& m, object o) {
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

static void set_input_div(bob::machine::SupportVector& m, object o) {
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

  class_<bob::machine::SVMFile, boost::shared_ptr<bob::machine::SVMFile>, boost::noncopyable>("SVMFile", "Loads a given libsvm data file. The data file format, as defined on the library README is like this:\n\n  <label> <index1>:<value1> <index2>:<value2> ...\n  .\n  .\n  .\n\nThe labels are integer values, so are the indexes, starting from '1' (and not from zero as a C-programmer would expect. The values are floating point.\n\nZero values are suppressed - libsvm uses a sparse format.\n\nThis class is made available to you so you can input original libsvm files and convert them to another representation better supported. You cannot, from this object, save data or extend the current set.", init<const char*>((arg("filename")), "Intializes an SVM file with the path to an existing file. The file is scanned entirely so to compute the sample size."))
    .add_property("shape", &svmfile_shape, "The size of each sample in the file, as tuple with a single entry")
    .add_property("__len__", &bob::machine::SVMFile::samples, "The size of each sample in the file, as tuple with a single entry")
    .add_property("filename", make_function(&bob::machine::SVMFile::filename, return_value_policy<copy_const_reference>()), "The name of the file being read")
    .def("reset", &bob::machine::SVMFile::reset, (arg("self")), "Resets the current file so it starts reading from the begin once more")
    .def("read", &svmfile_read, (arg("self")), "Reads a single line from the file and returns a tuple containing the label and a numpy array of float64 elements. The numpy array has a shape as defined by the 'shape' property of this file. If the file has finished, returns None instead.")
    .def("read", &svmfile_read2, (arg("self"), arg("values")), "Reads a single line from the file, but in this variant you should pass the numpy array where the results of the readout will be stored. The array should have a shape that is identical to what is returned by the 'shape' property of this file or an error will be raised. Returns the label for the entry. If the file is finished, returns None instead.")
    .def("read_", &svmfile_read2_, (arg("self"), arg("values")), "Reads a single line from the file and stores the contents of the line in the input array. Does not check the array shape. Returns the label for the entry. If the file is finished, returns None instead.")
    .def("read_all", &svmfile_read_all, (arg("self")), "Reads all remaining entries in the file. Returns a tuple containing (label, values) tuples for each entry in the file, in order of appearance.")
    .def("good", &bob::machine::SVMFile::good, (arg("self")), "Returns if the file is in a good state for readout. That means it has neither the eof, fail or bad bits set.")
    .def("fail", &bob::machine::SVMFile::fail, (arg("self")), "Tells if the file has the fail or bad bit set")
    .def("eof", &bob::machine::SVMFile::eof, (arg("self")), "Tells if the file has the eof bit set")
    ;

  enum_<bob::machine::SupportVector::svm_t>("svm_type")
    .value("C_SVC", bob::machine::SupportVector::C_SVC)
    .value("NU_SVC", bob::machine::SupportVector::NU_SVC)
    .value("ONE_CLASS", bob::machine::SupportVector::ONE_CLASS)
    .value("EPSILON_SVR", bob::machine::SupportVector::EPSILON_SVR)
    .value("NU_SVR", bob::machine::SupportVector::NU_SVR)
    ;

  enum_<bob::machine::SupportVector::kernel_t>("svm_kernel_type")
    .value("LINEAR", bob::machine::SupportVector::LINEAR)
    .value("POLY", bob::machine::SupportVector::POLY)
    .value("RBF", bob::machine::SupportVector::RBF)
    .value("SIGMOID", bob::machine::SupportVector::SIGMOID)
    .value("PRECOMPUTED", bob::machine::SupportVector::PRECOMPUTED)
    ;

  class_<bob::machine::SupportVector, boost::shared_ptr<bob::machine::SupportVector>, boost::noncopyable>("SupportVector", "This class can load and run an SVM generated by libsvm. Libsvm is a simple, easy-to-use, and efficient software for SVM classification and regression. It solves C-SVM classification, nu-SVM classification, one-class-SVM, epsilon-SVM regression, and nu-SVM regression. It also provides an automatic model selection tool for C-SVM classification. More information about libsvm can be found on its `website <http://www.csie.ntu.edu.tw/~cjlin/libsvm/>`_. In particular, this class covers most of the functionality provided by the command-line utility svm-predict.", no_init)
    .def(init<const char*>((arg("filename")), "Builds a new Support Vector Machine from a libsvm model file\n\nWhen you load using the libsvm model loader, note that the scaling parameters will be set to defaults (subtraction of 0.0 and division by 1.0). If you need scaling to be applied, set it individually using the appropriate methods bellow."))
    .def(init<bob::io::HDF5File&>((arg("config")), "Builds a new Support Vector Machine from an HDF5 file containing the configuration for this machine. Scaling parameters are also loaded from the file. Using this constructor assures a 100% state recovery from previous sessions."))
    .add_property("input_subtract", make_function(&bob::machine::SupportVector::getInputSubtraction, return_value_policy<copy_const_reference>()), &set_input_sub)
    .add_property("input_divide", make_function(&bob::machine::SupportVector::getInputDivision, return_value_policy<copy_const_reference>()), &set_input_div)
    .add_property("shape", &get_shape, "Expected input and output sizes. Note the output size is always 1.") 
    .add_property("labels", &labels, "The labels this machine will output.")
    .add_property("svm_type", &bob::machine::SupportVector::machineType, "The type of SVM machine contained")
    .add_property("kernel_type", &bob::machine::SupportVector::kernelType, "The type of kernel used by the support vectors in this machine")
    .add_property("degree", &bob::machine::SupportVector::polynomialDegree, "The polinomial degree, only valid if the kernel is polynomial")
    .add_property("gamma", &bob::machine::SupportVector::gamma, "The gamma parameter for polynomial, RBF (gaussian) or sigmoidal kernels")
    .add_property("coef0", &bob::machine::SupportVector::coefficient0, "The coefficient 0 for polynomial or sigmoidal kernels")
    .add_property("probability", &bob::machine::SupportVector::supportsProbability, "true if this machine supports probability outputs")
    .def("predict_class", &predict_class, (arg("self"), arg("input")), "Returns the predicted class given a certain input. Checks the input data for size conformity. If the size is wrong, an exception is raised.")
    .def("predict_class_", &predict_class_, (arg("self"), arg("input")), "Returns the predicted class given a certain input. Does not check the input data and is, therefore, a little bit faster.")
    .def("predict_classes", &predict_class_n, (arg("self"), arg("input")), "Returns the predicted class given a certain input. Checks the input data for size conformity. If the size is wrong, an exception is raised. This variant accepts as input a 2D array with samples arranged in lines. The array can have as many lines as you want, but the number of columns should match the expected machine input size.")
    .def("__call__", &svm_call, (arg("self"), arg("input")), "Returns the predicted class(es) given a certain input. Checks the input data for size conformity. If the size is wrong, an exception is raised. The input may be either a 1D or a 2D numpy ndarray object of double-precision floating-point numbers. If the array is 1D, a single answer is returned (the class of the input vector). If the array is 2D, then the number of columns in such array must match the input size. In this case, the SupportVector object will return 1 prediction for every row at the input array.")
    .def("predict_class_and_scores", &predict_class_and_scores2, (arg("self"), arg("input")), "Returns the predicted class and output scores as a tuple, in this order. Checks the input and output arrays for size conformity. If the size is wrong, an exception is raised.")
    .def("predict_class_and_scores", &predict_class_and_scores, (arg("self"), arg("input"), arg("scores")), "Returns the predicted class given a certain input. Returns the scores for each class in the second argument. Checks the input and output arrays for size conformity. If the size is wrong, an exception is raised.")
    .def("predict_class_and_scores_", &predict_class_and_scores_, (arg("self"), arg("input"), arg("scores")), "Returns the predicted class given a certain input. Returns the scores for each class in the second argument. Checks the input and output arrays for size conformity. Does not check the input data and is, therefore, a little bit faster.")
    .def("predict_classes_and_scores", &predict_class_and_scores_n, (arg("self"), arg("input")), "Returns the predicted class and output scores as a tuple, in this order. Checks the input array for size conformity. If the size is wrong, an exception is raised. This variant takes a single 2D double array as input. The samples should be organized row-wise.")
    .def("predict_class_and_probabilities", &predict_class_and_probs2, (arg("self"), arg("input")), "Returns the predicted class and probabilities in a tuple (on that order) given a certain input. The current machine has to support probabilities, otherwise an exception is raised. Checks the input array for size conformity. If the size is wrong, an exception is raised.")
    .def("predict_class_and_probabilities", &predict_class_and_probs, (arg("self"), arg("input"), arg("probabilities")), "Returns the predicted class given a certain input. If the model supports it, returns the probabilities for each class in the second argument, otherwise raises an exception. Checks the input and output arrays for size conformity. If the size is wrong, an exception is raised.")
    .def("predict_class_and_probabilities_", &predict_class_and_probs_, (arg("self"), arg("input"), arg("probabilities")), "Returns the predicted class given a certain input. This version will not run any checks, so you must be sure to pass the correct input to the classifier.")
    .def("predict_classes_and_probabilities", &predict_class_and_probs_n, (arg("self"), arg("input")), "Returns the predicted class and output probabilities for each possible class as a tuple, in this order. Checks the input array for size conformity. If the size is wrong, an exception is raised. This variant takes a single 2D double array as input. The samples should be organized row-wise.")
    .def("save", (void (bob::machine::SupportVector::*)(const std::string&) const)&bob::machine::SupportVector::save, (arg("self"), arg("filename")), "Saves the currently loaded model to an output file. Overwrites the file, if necessary")
    .def("save", (void (bob::machine::SupportVector::*)(bob::io::HDF5File&) const)&bob::machine::SupportVector::save, (arg("self"), arg("config")), "Saves the whole machine into a configuration file. This allows for a single instruction parameter loading, which includes both the model and the scaling parameters.")
    ;
}
