/**
 * @file cxx/trainer/src/SVMTrainer.cc
 * @date Sun  4 Mar 10:02:45 2012 CET
 * @author Andre Anjos <andre.anjos@idiap.ch>
 *
 * @brief Implementation of the SVM training methods
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

#include <boost/format.hpp>
#include "trainer/Exception.h"
#include "trainer/SVMTrainer.h"

namespace trainer = bob::trainer;

trainer::SVMTrainer::SVMTrainer(
    bob::machine::SupportVector::svm_t svm_type,
    bob::machine::SupportVector::kernel_t kernel_type,
    int degree,
    double gamma,
    double coef0,
    double cache_size,
    double eps,
    double C,
    double nu,
    double p,
    bool shrinking,
    bool probability
    )
{
  m_param.svm_type = svm_type;
  m_param.kernel_type = kernel_type;
  m_param.degree = degree;
  m_param.gamma = gamma;
  m_param.coef0 = coef0;
  m_param.cache_size = cache_size;
  m_param.eps = eps;
  m_param.C = C;
  m_param.nu = nu;
  m_param.p = p;
  m_param.shrinking = shrinking;
  m_param.probability = probability;

  //extracted from the data
  m_param.nr_weight = 0;
  m_param.weight_label = 0;
  m_param.weight = 0;
}

trainer::SVMTrainer::~SVMTrainer() { }

/**
 * Erases an SVM problem:
 *
 * struct svm_problem {
 *   int l; //number of entries
 *   double* y; //labels
 *   svm_node** x; //each set terminated with a -1 index entry
 * };
 *
 * At svm-train the nodes for each entry are allocated globally, what is
 * probably more efficient from the allocation perspective. It still requires
 * libsvm to scan the data twice to understand how many nodes need to be
 * allocated globally.
 */
void delete_problem(svm_problem* p) {
  delete[] p->y; //all labels
  for (int k=0; k<p->l; ++k) delete[] p->x[k];
  delete[] p->x;
  delete p;
}

/**
 * Allocates an svm_problem matrix
 */
svm_problem* new_problem(size_t entries) {
  svm_problem* retval = new(svm_problem);
  retval->l = (int)entries;
  retval->y = new double[entries];
  typedef svm_node* svm_node_ptr;
  retval->x = new svm_node_ptr[entries];
  for (size_t k=0; k<entries; ++k) retval->x[k] = 0;
  return retval;
}

/**
 * Converts the input arrayset data into an svm_problem matrix, used by libsvm
 * training routines. Updates "gamma" at the svm_parameter's.
 */
static boost::shared_ptr<svm_problem> data2problem
(const std::vector<bob::io::Arrayset>& data,
 const blitz::Array<double,1>& sub, const blitz::Array<double,1>& div,
 svm_parameter& param) {
  size_t entries = 0;
  for (size_t k=0; k<data.size(); ++k) entries += data[k].size();

  //allocates the container that will represent the problem; at this stage, we
  //allocate entries for each vector, but not the space in which feature will
  //be put at. This will come next.
  boost::shared_ptr<svm_problem> retval(new_problem(entries),
      std::ptr_fun(delete_problem));

  //choose labels.
  if (data.size() <= 1 | data.size() > 16) {
    boost::format m("Only supports SVMs for binary or multi-class classification problems (up to 16 classes). You passed me a list of %d arraysets.");
    m % data.size();
    throw std::runtime_error(m.str().c_str());
  }

  std::vector<double> labels;
  labels.reserve(data.size());
  if (data.size() == 2) {
    labels.push_back(-1.);
    labels.push_back(+1.);
  }
  else { //data.size() == 3, 4, ..., 16
    for (size_t k=0; k<data.size(); ++k) labels.push_back(k+1);
  }

  //iterate over each class data and instantiates the svm_nodes
  int max_index = 0; //data width

  size_t sample = 0;
  for (size_t k=0; k<data.size(); ++k) {
    for (size_t i=0; i<data[k].size(); ++k) {
      blitz::Array<double,1> d = data[k].get<double,1>(i);
      d = (d-sub)/div;
      size_t node = 1; //at least the "-1"-index terminator
      for (size_t p=0; p<d.size(); ++p) if (d(p)) ++node;
      retval->x[i] = new svm_node[node];
      node = 0;
      for (size_t p=0; p<d.size(); ++p) {
        if (d(p)) {
          int index = p+1; //starts indexing at 1
          retval->y[sample] = labels[k];
          retval->x[sample][node].index = index;
          retval->x[sample][node].value = d(p);
          if ( index > max_index ) max_index = index;
          ++node; //index within the current sample
        }
      }
      //marks end of sequence
      retval->x[sample][node].index = -1.;
      retval->x[sample][node].value = 0;
      ++sample;
    }
  }

  //extracted from svm-train.c
  if (param.gamma == 0. && max_index > 0) {
    param.gamma = 1.0/max_index;
  }

  //do not support pre-computed kernels...
  if (param.kernel_type == PRECOMPUTED) {
    throw std::runtime_error("We currently dod not support PRECOMPUTED kernels in these bindings to libsvm");
  }

  return retval;
}

/**
 * A wrapper, to standardize the freeing of the svm_model
 */
static void svm_model_free(svm_model*& m) {
#if LIBSVM_VERSION >= 300
  svm_free_and_destroy_model(&m);
#else
  svm_destroy_model(m);
#endif
}

boost::shared_ptr<bob::machine::SupportVector> trainer::SVMTrainer::train
(const std::vector<bob::io::Arrayset>& data,
 const blitz::Array<double,1>& input_subtraction,
 const blitz::Array<double,1>& input_division) const {
  //sanity check of input arraysets
  int n_features = data[0].getShape()[0];
  for (size_t cl=0; cl<data.size(); ++cl) {
    if (data[cl].getElementType() != bob::core::array::t_float64) {
      throw io::TypeError(data[cl].getElementType(),
          bob::core::array::t_float64);
    }
    if (data[cl].getNDim() != 1) {
      throw bob::io::DimensionError(data[cl].getNDim(), 1);
    }
    if (data[cl].getShape()[0] != (size_t)n_features) {
      throw bob::trainer::WrongNumberOfFeatures(data[cl].getShape()[0],
          n_features, cl);
    }
  }

  //converts the input arraysets into something libsvm can digest
  double save_gamma = m_param.gamma; ///< the next method may update it!
  boost::shared_ptr<svm_problem> problem = 
    data2problem(data, input_subtraction, input_division, 
        const_cast<svm_parameter&>(m_param) ///< temporary cast
        );
  
  //checks parametrization to make sure all is alright.
  const char* error_msg = svm_check_parameter(problem.get(), &m_param);

  if (error_msg) {
    const_cast<double&>(m_param.gamma) = save_gamma;
    boost::format m("libsvm-%d reports: %s");
    m % LIBSVM_VERSION % error_msg;
    std::runtime_error(m.str().c_str());
  }

  //do the training, returns the new machine
  boost::shared_ptr<svm_model> model(svm_train(problem.get(), &m_param),
      std::ptr_fun(svm_model_free));

  const_cast<double&>(m_param.gamma) = save_gamma;

  boost::shared_ptr<bob::machine::SupportVector> retval =
    boost::make_shared<bob::machine::SupportVector>(model);

  //sets up the scaling parameters given as input
  retval->setInputSubtraction(input_subtraction);
  retval->setInputDivision(input_division);

  return retval;
}

boost::shared_ptr<bob::machine::SupportVector> trainer::SVMTrainer::train
(const std::vector<bob::io::Arrayset>& data) const {
  int n_features = data[0].getShape()[0];
  blitz::Array<double,1> sub(n_features);
  sub = 0.;
  blitz::Array<double,1> div(n_features);
  div = 1.;
  return train(data, sub, div);
}
