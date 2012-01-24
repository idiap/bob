/**
 * @file cxx/measure/src/GaborJetSimilarities.cc
 * @date
 * @author
 *
 * @brief Implements the Gabor jet similarity functions
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

#include "measure/GaborJetSimilarities.h"

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////  Simple similarity functions  //////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////

double bob::measure::ScalarProductSimilarity::similarity(const blitz::Array<double,2>& jet1, const blitz::Array<double,2>& jet2) const{
  const blitz::Array<double,1> j1 = jet1(0,blitz::Range::all()), j2 = jet2(0,blitz::Range::all());
  bob::core::array::assertCZeroBaseContiguous(j1);
  bob::core::array::assertCZeroBaseContiguous(j2);
  bob::core::array::assertSameShape(j1,j2);
  return std::inner_product(j1.begin(), j1.end(), j2.begin(), 0.);
}

double bob::measure::CanberraSimilarity::similarity(const blitz::Array<double,2>& jet1, const blitz::Array<double,2>& jet2) const{
  const blitz::Array<double,1> j1 = jet1(0,blitz::Range::all()), j2 = jet2(0,blitz::Range::all());
  bob::core::array::assertCZeroBaseContiguous(j1);
  bob::core::array::assertCZeroBaseContiguous(j2);
  bob::core::array::assertSameShape(j1,j2);
  double sim = 0.;
  unsigned size = j1.shape()[0];
  for (unsigned j = size; j--;){
    sim += 1. - std::abs(j1(j) - j2(j)) / (j1(j) + j2(j));
  }
  return sim / size;
}


/////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////  Disparity estimation  /////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////
static double sqr(double x){return x*x;}

static double adjustPhase(double phase){
  return phase - (2.*M_PI)*round(phase / (2.*M_PI));
}

bob::measure::DisparitySimilarity::DisparitySimilarity(const bob::ip::GaborWaveletTransform& gwt)
  : GaborJetSimilarity(),
    m_kernel_frequencies(gwt.kernelFrequencies()),
    m_number_of_scales(gwt.m_number_of_scales),
    m_number_of_directions(gwt.m_number_of_directions),
    m_disparity(0.,0.),
    m_confidences(m_number_of_scales * m_number_of_directions, 0.),
    m_phase_differences(m_number_of_scales * m_number_of_directions, 0.)
{
  m_wavelet_extends.reserve(m_number_of_scales);
  for (int level = 0; level < m_number_of_scales; ++level){
    double k = sqrt(sqr(m_kernel_frequencies[level * m_number_of_directions].first) + sqr(m_kernel_frequencies[level * m_number_of_directions].second));
    m_wavelet_extends.push_back(M_PI / k);
  }
}

double bob::measure::DisparitySimilarity::similarity(const blitz::Array<double,2>& jet1, const blitz::Array<double,2>& jet2) const{
  // first, fill confidence and phase difference vectors
  for (int j = m_confidences.size(); j--;){
    m_confidences[j] = jet1(0,j) * jet2(0,j);
    m_phase_differences[j] = adjustPhase(jet1(1,j) - jet2(1,j));
  }

  // now, compute the disparity
  // approximate the disparity from the phase differences
  double gamma_x_x = 0., gamma_x_y = 0., gamma_y_y = 0., phi_x = 0., phi_y = 0.;
  // initialize the disparity with 0
  m_disparity = std::make_pair(0.,0.);

  // iterate backwards through the vector to start with the lowest frequency wavelets
  for (int j = m_confidences.size(), level = m_number_of_scales-1; level >= 0; --level){
    for (int direction = m_number_of_directions-1; direction >= 0; --direction, --j){
      double kjx = m_kernel_frequencies[j].first,
          kjy = m_kernel_frequencies[j].second,
          conf = m_confidences[j],
          diff = m_phase_differences[j];

      // totalize gamma matrix
      gamma_x_x += kjx * kjx * conf;
      gamma_x_y += kjx * kjy * conf;
      gamma_y_y += kjy * kjy * conf;

      // totalize phi vector
      // estimate the number of cycles that we are off
      double nL = round((diff - m_disparity.first * kjx - m_disparity.second * kjy) / (2.*M_PI));
      // totalize corrected phi vector elements
      phi_x += (diff - nL * 2. * M_PI) * conf * kjx;
      phi_y += (diff - nL * 2. * M_PI) * conf * kjy;
    } // for direction

    // re-calculate disparity as d=\Gamma^{-1}\Phi of the (low frequency) wavelet scales that we used up to now
    double gamma_det = gamma_x_x * gamma_y_y - sqr(gamma_x_y);
    m_disparity.first  = (gamma_y_y * phi_x - gamma_x_y * phi_y) / gamma_det;
    m_disparity.second = (gamma_x_x * phi_y - gamma_x_y * phi_x) / gamma_det;

  } // for level


  // finally, compute the similarity using the estimated disparity
  double sum = 0.;
  for (int j = m_confidences.size(); j--;){
    sum += m_confidences[j] * cos(m_phase_differences[j] - m_disparity.first * m_kernel_frequencies[j].first - m_disparity.second * m_kernel_frequencies[j].second);
  }

  return sum;
}
