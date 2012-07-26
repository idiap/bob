#include <numeric>
#include <functional>
#include <fstream>

#include "visioner/util/histogram.h"

namespace bob { namespace visioner {

  // Constructor
  Histogram::Histogram(index_t n_bins, scalar_t min_value, scalar_t max_value)
  {
    reset(n_bins, min_value, max_value);
  }

  // Reset to the new bins
  void Histogram::reset(index_t n_bins, scalar_t min_value, scalar_t max_value)
  {
    m_min = std::min(min_value, max_value);
    m_max = std::max(min_value, max_value);
    m_n_bins = std::max(n_bins, (index_t)1);
    m_delta = (m_max - m_min) / m_n_bins;
    m_inv_delta = inverse(m_delta);

    m_bins.resize(m_n_bins);
    clear();
  }

  // Add a new value
  void Histogram::add(scalar_t value)
  {
    value = range(value, m_min, m_max);
    m_bins[range((int)((value - m_min) * m_inv_delta), 0, m_n_bins - 1)] ++;
  }

  // Compute and normalize the cumulated histogram
  void Histogram::cumulate()
  {
    std::partial_sum(m_bins.begin(), m_bins.end(), m_bins.begin());

    std::transform(	m_bins.begin(), m_bins.end(), m_bins.begin(),
        std::bind2nd(std::multiplies<scalar_t>(), inverse(m_bins[m_n_bins - 1])));
  }

  // Normalize the histogram
  void Histogram::norm()
  {
    const scalar_t sum = std::accumulate(m_bins.begin(), m_bins.end(), 0.0);                
    std::transform(	m_bins.begin(), m_bins.end(), m_bins.begin(),
        std::bind2nd(std::multiplies<scalar_t>(), inverse(sum)));
  }

  // Save to file
  bool Histogram::save(const string_t& path) const
  {
    std::ofstream out(path.c_str());
    if (out.is_open() == false)
    {
      return false;
    }

    for (index_t i = 0; i < n_bins(); i ++)
    {
      out << bin_value(i) << "\t" << bins()[i] << "\n";
    }

    return true;
  }

}}
