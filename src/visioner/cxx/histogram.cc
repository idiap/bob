/**
 * @file visioner/cxx/histogram.cc
 * @date Fri 27 Jul 13:58:57 2012 CEST
 * @author Andre Anjos <andre.anjos@idiap.ch>
 *
 * @brief This file was part of Visioner and originally authored by "Cosmin
 * Atanasoaei <cosmin.atanasoaei@idiap.ch>". It was only modified to conform to
 * Bob coding standards and structure.
 *
 * Copyright (C) 2011-2013 Idiap Research Institute, Martigny, Switzerland
 */

#include <numeric>
#include <functional>
#include <fstream>

#include "bob/visioner/util/histogram.h"

namespace bob { namespace visioner {

  // Constructor
  Histogram::Histogram(uint64_t n_bins, double min_value, double max_value)
  {
    reset(n_bins, min_value, max_value);
  }

  // Reset to the new bins
  void Histogram::reset(uint64_t n_bins, double min_value, double max_value)
  {
    m_min = std::min(min_value, max_value);
    m_max = std::max(min_value, max_value);
    m_n_bins = std::max(n_bins, (uint64_t)1);
    m_delta = (m_max - m_min) / m_n_bins;
    m_inv_delta = inverse(m_delta);

    m_bins.resize(m_n_bins);
    clear();
  }

  // Add a new value
  void Histogram::add(double value)
  {
    value = range(value, m_min, m_max);
    m_bins[range((int)((value - m_min) * m_inv_delta), 0, m_n_bins - 1)] ++;
  }

  // Compute and normalize the cumulated histogram
  void Histogram::cumulate()
  {
    std::partial_sum(m_bins.begin(), m_bins.end(), m_bins.begin());

    std::transform(	m_bins.begin(), m_bins.end(), m_bins.begin(),
        std::bind2nd(std::multiplies<double>(), inverse(m_bins[m_n_bins - 1])));
  }

  // Normalize the histogram
  void Histogram::norm()
  {
    const double sum = std::accumulate(m_bins.begin(), m_bins.end(), 0.0);                
    std::transform(	m_bins.begin(), m_bins.end(), m_bins.begin(),
        std::bind2nd(std::multiplies<double>(), inverse(sum)));
  }

  // Save to file
  bool Histogram::save(const std::string& path) const
  {
    std::ofstream out(path.c_str());
    if (out.is_open() == false)
    {
      return false;
    }

    for (uint64_t i = 0; i < n_bins(); i ++)
    {
      out << bin_value(i) << "\t" << bins()[i] << "\n";
    }

    return true;
  }

}}
