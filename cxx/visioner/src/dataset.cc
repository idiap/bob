#include "visioner/model/dataset.h"

namespace bob { namespace visioner {

  // Constructor
  DataSet::DataSet(index_t n_outputs, index_t n_samples, index_t n_features, index_t n_fvalues)
    :	m_n_fvalues(n_fvalues)
  {
    resize(n_outputs, n_samples, n_features, n_fvalues);
  }

  // Resize
  void DataSet::resize(index_t n_outputs, index_t n_samples, index_t n_features, index_t n_fvalues)
  {
    m_n_fvalues = n_fvalues;
    m_targets.resize(n_samples, n_outputs);
    m_values.resize(n_features, n_samples);
    m_costs.resize(n_samples);
  }

}}
