/**
 * @file cxx/machine/src/SVM.cc
 * @date Sat Dec 17 14:41:56 2011 +0100
 * @author Andre Anjos <andre.anjos@idiap.ch>
 *
 * @brief Implementation of the SVM machine using libsvm
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

#include <string>
#include <cmath>
#include <boost/format.hpp>
#include <boost/algorithm/string.hpp>
#include <boost/filesystem.hpp>
#include "machine/SVM.h"
#include "machine/MLPException.h"
#include "core/array_check.h"
#include "core/logging.h"
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sys/types.h>
#include <sys/stat.h>

namespace mach = bob::machine;
namespace array = bob::core::array;

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
static void svm_model_free(svm_model*& m) {
#if LIBSVM_VERSION >= 300
  svm_free_and_destroy_model(&m);
#else
  svm_destroy_model(m);
#endif
}

blitz::Array<uint8_t,1> mach::svm_pickle
(const boost::shared_ptr<svm_model> model)
{
  //use a re-entrant version of tmpnam...
  char tmp_filename[L_tmpnam]; 
  char* v = std::tmpnam(tmp_filename);
  if (!v) throw std::runtime_error("std::tmpnam() call failed - unique name cannot be generated");

  //save it to a temporary file
  if (svm_save_model(tmp_filename, model.get())) {
    boost::format s("cannot save SVM to file `%s' while copying model");
    s % tmp_filename;
    throw std::runtime_error(s.str().c_str());
  }

  //gets total size of file
  struct stat filestatus;
  stat(tmp_filename, &filestatus);
 
  //reload the data from the file in binary format
  std::ifstream binfile(tmp_filename, std::ios::binary);
  blitz::Array<uint8_t,1> buffer(filestatus.st_size);
  binfile.read(reinterpret_cast<char*>(buffer.data()), filestatus.st_size);

  //unlink the temporary file
  boost::filesystem::remove(tmp_filename); 

  //finally, return the pickled data
  return buffer;
}

/**
 * Reverts the pickling process, returns the model
 */
boost::shared_ptr<svm_model> mach::svm_unpickle
(const blitz::Array<uint8_t,1>& buffer) {
  //use a re-entrant version of tmpnam...
  char tmp_filename[L_tmpnam];
  char* v = std::tmpnam(tmp_filename);
  if (!v) throw std::runtime_error("std::tmpnam() call failed - unique name cannot be generated");

  std::ofstream binfile(tmp_filename, std::ios::binary);
  binfile.write(reinterpret_cast<const char*>(buffer.data()), buffer.size());
  binfile.close();

  //reload the file using the appropriate libsvm loading method
  boost::shared_ptr<svm_model> retval(svm_load_model(tmp_filename), 
      std::ptr_fun(svm_model_free));

  //unlinks the temporary file
  boost::filesystem::remove(tmp_filename); 

  //finally, return the pickled data
  return retval;
}

void mach::SupportVector::reset() {
  //gets the expected size for the input from the SVM
  m_input_size = 0;
  for (int k=0; k<m_model->l; ++k) {
    svm_node* end = m_model->SV[k];
    while (end->index != -1) {
      if (end->index > (int)m_input_size) m_input_size = end->index;
      ++end;
    }
  }

  //create and reset cache
  m_input_cache.reset(new svm_node[1 + m_model->l]);
  for (int k=0; k<m_model->l; ++k) {
    m_input_cache[k].index = k+1;
    m_input_cache[k].value = 0.;
  }
  m_input_cache[m_model->l].index = -1; //libsvm detects end of input if idx=-1

  m_input_sub.resize(inputSize());
  m_input_sub = 0.0;
  m_input_div.resize(inputSize());
  m_input_div = 1.0;
}

mach::SupportVector::SupportVector(const std::string& model_file):
  m_model(svm_load_model(model_file.c_str()), std::ptr_fun(svm_model_free))
{
  if (!m_model) {
    boost::format s("cannot open model file '%s'");
    s % model_file;
    throw std::runtime_error(s.str().c_str());
  }
  reset();
}

mach::SupportVector::SupportVector(bob::io::HDF5File& config):
  m_model()
{
  if ( (LIBSVM_VERSION/100) > (config.getVersion()/100) ) {
    //if the major version changes... be aware!
    boost::format m("SVM being loaded from `%s:%s' (created with libsvm-%d) with libsvm-%d. You may want to read the libsvm FAQ at http://www.csie.ntu.edu.tw/~cjlin/libsvm/log to check if there were format changes between these versions. If not, you can safely ignore this warning and even tell us to remove it via our bug tracker: https://github.com/idiap/bob/issues");
    m % config.filename() % config.cwd() % config.getVersion() % LIBSVM_VERSION;
    bob::core::warn << m.str() << std::endl;
  }
  m_model = mach::svm_unpickle(config.readArray<uint8_t,1>("svm_model"));
  reset(); ///< note: has to be done before reading scaling parameters
  config.readArray("input_subtract", m_input_sub);
  config.readArray("input_divide", m_input_div);
}

mach::SupportVector::SupportVector(boost::shared_ptr<svm_model> model)
  : m_model(model)
{
  if (!m_model) {
    throw std::runtime_error("null SVM model cannot be processed");
  }
  reset();
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

void mach::SupportVector::setInputSubtraction(const blitz::Array<double,1>& v) {
  if (inputSize() != (size_t)v.extent(0)) {
    throw mach::NInputsMismatch(inputSize(), v.extent(0));
  }
  m_input_sub.reference(bob::core::array::ccopy(v));
}

void mach::SupportVector::setInputDivision(const blitz::Array<double,1>& v) {
  if (inputSize() != (size_t)v.extent(0)) {
    throw mach::NInputsMismatch(inputSize(), v.extent(0));
  }
  m_input_div.reference(bob::core::array::ccopy(v));
}

/**
 * Copies the user input to a locally pre-allocated cache. Apply normalization
 * at the same occasion.
 */
static inline void copy(const blitz::Array<double,1>& input,
    boost::shared_array<svm_node>& cache, const blitz::Array<double,1>& sub,
    const blitz::Array<double,1>& div) {
  for (size_t k=0; k<(size_t)input.extent(0); ++k) 
    cache[k].value = (input(k) - sub(k))/div(k);
}

int mach::SupportVector::predictClass_
(const blitz::Array<double,1>& input) const {
  copy(input, m_input_cache, m_input_sub, m_input_div);
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
  copy(input, m_input_cache, m_input_sub, m_input_div);
#if LIBSVM_VERSION > 290
  int retval = round(svm_predict_values(m_model.get(), m_input_cache.get(), scores.data()));
#else
  svm_predict_values(m_model.get(), m_input_cache.get(), scores.data());
  int retval = round(svm_predict(m_model.get(), m_input_cache.get()));
#endif
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
  copy(input, m_input_cache, m_input_sub, m_input_div);
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

void mach::SupportVector::save(const std::string& filename) const {
  if (svm_save_model(filename.c_str(), m_model.get())) {
    boost::format s("cannot save SVM model to file '%s'");
    s % filename;
    throw std::runtime_error(s.str().c_str());
  }
}

void mach::SupportVector::save(bob::io::HDF5File& config) const {
  config.setArray("svm_model", mach::svm_pickle(m_model));
  config.setArray("input_subtract", m_input_sub);
  config.setArray("input_divide", m_input_div);
  config.setVersion(LIBSVM_VERSION);
}
