/**
 * @author Andre Anjos <andre.anjos@idiap.ch>
 * @date Fri  2 Dec 18:10:24 2011
 *
 * @brief Implementation of the SVM machine using libsvm
 *
 * Copyright (C) 2011 Idiap Reasearch Institute, Martigny, Switzerland
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

#include <string>
#include <cmath>
#include <boost/format.hpp>
#include <boost/algorithm/string.hpp>
#include "machine/SVM.h"
#include "core/array_check.h"

// We need to declare the svm_model type for libsvm < 3.0.0. The next bit of
// code was cut and pasted from version 2.9.1 of libsvm, file svm.cpp.
#if LIBSVM_VERSION < 300
struct svm_model {
	struct svm_parameter param;	/* parameter */
	int nr_class;		      /* number of classes, = 2 in regression/one class svm */
	int l;			          /* total #SV */
	struct svm_node **SV;	/* SVs (SV[l]) */
	double **sv_coef;	    /* coefficients for SVs in decision functions (sv_coef[k-1][l]) */
	double *rho;		      /* constants in decision functions (rho[k*(k-1)/2]) */
	double *probA;		    /* pariwise probability information */
	double *probB;

	/* for classification only */

	int *label;  /* label of each class (label[k]) */
	int *nSV;		 /* number of SVs for each class (nSV[k]) */
				       /* nSV[0] + nSV[1] + ... + nSV[k-1] = l */
	/* XXX */
	int free_sv; /* 1 if svm_model is created by svm_load_model*/
				       /* 0 if svm_model is created by svm_train */
};
#endif

namespace mach = Torch::machine;
namespace array = Torch::core::array;

mach::SVMFile::SVMFile (const std::string& filename, size_t shape):
  m_filename(filename),
  m_file(m_filename.c_str()),
  m_shape(shape)
{
  if (!m_file) {
    boost::format s("cannot open file '%s'");
    s % filename;
    throw std::runtime_error(s.str().c_str());
  }
}

mach::SVMFile::~SVMFile() {
}

void mach::SVMFile::reset() {
  m_file.close();
  m_file.open(m_filename.c_str());
}

bool mach::SVMFile::read(int& label, blitz::Array<double,1>& values) {
  if ((size_t)values.extent(0) != m_shape) {
    boost::format s("file '%s' contains %d entries per sample, but you gave me an array with only %d positions");
    s % m_filename % m_shape % values.extent(0);
    throw std::invalid_argument(s.str().c_str());
  }

  //read the data.
  return read_(label, values);
}

bool mach::SVMFile::read_(int& label, blitz::Array<double,1>& values) {
  
  //if the file is at the end, just raise, you should have checked
  if (!m_file.good()) return false;

  //gets the next non-empty line
  std::string line;
  while (!line.size()) {
    if (!m_file.good()) return false;
    std::getline(m_file, line);
    boost::trim(line);
  }

  std::istringstream iss(line);
  iss >> label;

  int pos;
  char separator;
  double value;

  values = 0; ///zero values all over as the data is sparse on the files

  for (size_t k=0; k<m_shape; ++k) {
    iss >> pos >> separator >> value;
    values(pos-1) = value;
  }

  return true;
}

/**
 * A wrapper, to standardize this function.
 */
static void my_model_free(svm_model*& m) {
#if LIBSVM_VERSION >= 300
  svm_free_and_destroy_model(&m);
#else
  svm_destroy_model(&m);
#endif
}

mach::SupportVector::SupportVector(const char* model_file):
  m_model(svm_load_model(model_file), std::ptr_fun(my_model_free))
{
  if (!m_model) {
    boost::format s("cannot open model file %s");
    s % model_file;
    throw std::runtime_error(s.str().c_str());
  }

  //gets the expected size for the input from the first SVM
  svm_node* end = m_model->SV[0];
  while (end->index != -1) end += 1;
  m_input_size = (size_t)(end - m_model->SV[0]);

  //create and reset cache
  m_input_cache.reset(new svm_node[1 + m_model->l]);
  for (int k=0; k<m_model->l; ++k) {
    m_input_cache[k].index = k+1;
    m_input_cache[k].value = 0.;
  }
  m_input_cache[m_model->l].index = -1; //libsvm detects end of input if idx=-1
}

mach::SupportVector::~SupportVector() { }

bool mach::SupportVector::supportsProbability() const {
  return svm_check_probability_model(m_model.get());
}

size_t mach::SupportVector::inputSize() const {
  return m_input_size;
}

size_t mach::SupportVector::outputSize() const {
  size_t retval = svm_get_nr_class(m_model.get());
  return (retval == 2)? 1 : retval;
}

size_t mach::SupportVector::numberOfClasses() const {
  return svm_get_nr_class(m_model.get());
}

int mach::SupportVector::classLabel(size_t i) const {

  if (i >= (size_t)svm_get_nr_class(m_model.get())) {
    boost::format s("request for label of class %d in SVM with %d classes is not legal");
    s % (int)i % svm_get_nr_class(m_model.get());
    throw std::invalid_argument(s.str().c_str());
  }
  return m_model->label[i];

}

mach::SupportVector::svm_t mach::SupportVector::machineType() const {
  return (svm_t)svm_get_svm_type(m_model.get());
}

mach::SupportVector::kernel_t mach::SupportVector::kernelType() const {
  return (kernel_t)m_model->param.kernel_type;
}

int mach::SupportVector::polynomialDegree() const {
  return m_model->param.degree;
}

double mach::SupportVector::gamma() const {
  return m_model->param.gamma;
}

double mach::SupportVector::coefficient0() const {
  return m_model->param.coef0;
}

/**
 * Copies the user input to a locally pre-allocated cache.
 */
static inline void copy(const blitz::Array<double,1>& input,
    boost::shared_array<svm_node>& cache) {
  for (size_t k=0; k<(size_t)input.extent(0); ++k) cache[k].value = input(k);
}

int mach::SupportVector::predictClass_
(const blitz::Array<double,1>& input) const {
  copy(input, m_input_cache);
  int retval = round(svm_predict(m_model.get(), m_input_cache.get()));
  return retval;
}

int mach::SupportVector::predictClass
(const blitz::Array<double,1>& input) const {

  if ((size_t)input.extent(0) != inputSize()) {
    boost::format s("input for this SVM should have %d components, but you provided an array with %d elements instead");
    s % inputSize() % input.extent(0);
    throw std::invalid_argument(s.str().c_str());
  }

  return predictClass_(input); 
}

int mach::SupportVector::predictClassAndScores_
(const blitz::Array<double,1>& input,
 blitz::Array<double,1>& scores) const {
  copy(input, m_input_cache);
  int retval = round(svm_predict_values(m_model.get(), m_input_cache.get(), scores.data()));
  return retval;
}

int mach::SupportVector::predictClassAndScores
(const blitz::Array<double,1>& input,
 blitz::Array<double,1>& scores) const {
   
  if ((size_t)input.extent(0) != inputSize()) {
    boost::format s("input for this SVM should have %d components, but you provided an array with %d elements instead");
    s % inputSize() % input.extent(0);
    throw std::invalid_argument(s.str().c_str());
  }

  if (!array::isCContiguous(scores)) {
    throw std::invalid_argument("scores output array should be C-style contiguous and what you provided is not");
  }

  if ((size_t)scores.extent(0) != outputSize()) {
    boost::format s("output scores for this SVM should have %d components, but you provided an array with %d elements instead");
    s % outputSize() % scores.extent(0);
    throw std::invalid_argument(s.str().c_str());
  }

  return predictClassAndScores_(input, scores);
}

int mach::SupportVector::predictClassAndProbabilities_
(const blitz::Array<double,1>& input,
 blitz::Array<double,1>& probabilities) const {
  copy(input, m_input_cache);
  int retval = round(svm_predict_probability(m_model.get(), m_input_cache.get(), probabilities.data()));
  return retval;
}

int mach::SupportVector::predictClassAndProbabilities
(const blitz::Array<double,1>& input,
 blitz::Array<double,1>& probabilities) const {
   
  if ((size_t)input.extent(0) != inputSize()) {
    boost::format s("input for this SVM should have %d components, but you provided an array with %d elements instead");
    s % inputSize() % input.extent(0);
    throw std::invalid_argument(s.str().c_str());
  }

  if (!supportsProbability()) {
    throw std::runtime_error("this SVM does not support probabilities");
  }

  if (!array::isCContiguous(probabilities)) {
    throw std::invalid_argument("probabilities output array should be C-style contiguous and what you provided is not");
  }

  if ((size_t)probabilities.extent(0) != outputSize()) {
    boost::format s("output probabilities for this SVM should have %d components, but you provided an array with %d elements instead");
    s % outputSize() % probabilities.extent(0);
    throw std::invalid_argument(s.str().c_str());
  }

  return predictClassAndProbabilities_(input, probabilities);
}

void mach::SupportVector::save(const char* filename) const {
  if (svm_save_model(filename, m_model.get())) {
    boost::format s("cannot save SVM model to file '%s'");
    s % filename;
    throw std::runtime_error(s.str().c_str());
  }
}
