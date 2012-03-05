/**
 * @file cxx/machine/src/GaborJetSimilarities.cc
 * @date 2012-03-05
 * @author Manuel Guenther <Manuel.Guenther@idiap.ch>
 *
 * @brief Implements the Gabor jet similarity functions.
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

#include "machine/GaborJetSimilarities.h"

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////  Simple similarity functions  //////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/**
 * This function computes the similarity between two Gabor jets as the
 * inner product of the absolute parts of the jets \f[ \sum_j a_j\,a_j' \f]
 * @param jet1 One of the two Gabor jets to compare
 * @param jet2 One of the two Gabor jets to compare
 * @return The similarity of jet1 and jet2, a value between 0 and 1
 */
double bob::machine::ScalarProductSimilarity::similarity(const blitz::Array<double,1>& jet1, const blitz::Array<double,1>& jet2) const{
  bob::core::array::assertCZeroBaseContiguous(jet1);
  bob::core::array::assertCZeroBaseContiguous(jet2);
  bob::core::array::assertSameShape(jet1,jet2);
  return std::inner_product(jet1.begin(), jet1.end(), jet2.begin(), 0.);
}

/**
 * This function computes the similarity between two Gabor jets as the
 * inner product of the absolute parts of the jets \f[ \sum_j a_j\,a_j' \f]
 * @param jet1 One of the two Gabor jets to compare
 * @param jet2 One of the two Gabor jets to compare
 * @return The similarity of jet1 and jet2, a value between 0 and 1
 */
double bob::machine::ScalarProductSimilarity::similarity(const blitz::Array<double,2>& jet1, const blitz::Array<double,2>& jet2) const{
  const blitz::Array<double,1> j1 = jet1(0,blitz::Range::all()), j2 = jet2(0,blitz::Range::all());
  return similarity(j1, j2);
}



/**
 * This function computes the similarity between two Gabor jets as the
 * inner product of the absolute parts of the jets
 * \f[ 1 - \frac1J \sum_j \frac{|a_j - a_j'|}{a_j + a_j'} \f]
 * @param jet1 One of the two Gabor jets to compare
 * @param jet2 One of the two Gabor jets to compare
 * @return The similarity of jet1 and jet2, a value between 0 and 1
 */
double bob::machine::CanberraSimilarity::similarity(const blitz::Array<double,1>& jet1, const blitz::Array<double,1>& jet2) const{
  bob::core::array::assertCZeroBaseContiguous(jet1);
  bob::core::array::assertCZeroBaseContiguous(jet2);
  bob::core::array::assertSameShape(jet1,jet2);
  double sim = 0.;
  unsigned size = jet1.shape()[0];
  for (unsigned j = size; j--;){
    sim += 1. - std::abs(jet1(j) - jet2(j)) / (jet1(j) + jet2(j));
  }
  return sim / size;
}

/**
 * This function computes the similarity between two Gabor jets as the
 * inner product of the absolute parts of the jets
 * \f[ 1 - \frac1J \sum_j \frac{|a_j - a_j'|}{a_j + a_j'} \f]
 * @param jet1 One of the two Gabor jets to compare
 * @param jet2 One of the two Gabor jets to compare
 * @return The similarity of jet1 and jet2, a value between 0 and 1
 */
double bob::machine::CanberraSimilarity::similarity(const blitz::Array<double,2>& jet1, const blitz::Array<double,2>& jet2) const{
  const blitz::Array<double,1> j1 = jet1(0,blitz::Range::all()), j2 = jet2(0,blitz::Range::all());
  return similarity(j1, j2);
}


/////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////  Disparity estimation  /////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////
static double sqr(double x){return x*x;}

static double adjustPhase(double phase){
  return phase - (2.*M_PI)*round(phase / (2.*M_PI));
}

/**
 * Constructor for the DisparitySimilarity function that requires the information about the
 * GaborWaveletTransform that was used to generate the Gabor jets that will be compared later on
 * @param gwt The GaborWaveletTransform class with which the Gbaor jets were generated
 */
bob::machine::DisparitySimilarity::DisparitySimilarity(const bob::ip::GaborWaveletTransform& gwt)
  : GaborJetSimilarity(),
    m_disparity(0.,0.),
    m_confidences(gwt.m_number_of_scales * gwt.m_number_of_directions, 0.),
    m_phase_differences(gwt.m_number_of_scales * gwt.m_number_of_directions, 0.),
    m_kernel_frequencies(gwt.kernelFrequencies()),
    m_number_of_scales(gwt.m_number_of_scales),
    m_number_of_directions(gwt.m_number_of_directions)
{
  m_wavelet_extends.reserve(m_number_of_scales);
  for (int level = 0; level < m_number_of_scales; ++level){
    double k = sqrt(sqr(m_kernel_frequencies[level * m_number_of_directions][0]) + sqr(m_kernel_frequencies[level * m_number_of_directions][1]));
    m_wavelet_extends.push_back(M_PI / k);
  }
}

void bob::machine::DisparitySimilarity::compute_confidences(const blitz::Array<double,2>& jet1, const blitz::Array<double,2>& jet2) const{
  // first, fill confidence and phase difference vectors
  for (int j = m_confidences.size(); j--;){
    m_confidences[j] = jet1(0,j) * jet2(0,j);
    m_phase_differences[j] = adjustPhase(jet1(1,j) - jet2(1,j));
  }
}

void bob::machine::DisparitySimilarity::compute_disparity() const{
  // approximate the disparity from the phase differences
  double gamma_x_x = 0., gamma_x_y = 0., gamma_y_y = 0., phi_x = 0., phi_y = 0.;
  // initialize the disparity with 0
  m_disparity = 0.;

  // iterate backwards through the vector to start with the lowest frequency wavelets
  for (int j = m_confidences.size(), level = m_number_of_scales-1; level >= 0; --level){
    for (int direction = m_number_of_directions-1; direction >= 0; --direction, --j){
      double
          kjx = m_kernel_frequencies[j][1],
          kjy = m_kernel_frequencies[j][0],
          conf = m_confidences[j],
          diff = m_phase_differences[j];

      // totalize gamma matrix
      gamma_x_x += kjx * kjx * conf;
      gamma_x_y += kjx * kjy * conf;
      gamma_y_y += kjy * kjy * conf;

      // totalize phi vector
      // estimate the number of cycles that we are off
      double nL = round((diff - m_disparity[1] * kjx - m_disparity[0] * kjy) / (2.*M_PI));
      // totalize corrected phi vector elements
      phi_x += (diff - nL * 2. * M_PI) * conf * kjx;
      phi_y += (diff - nL * 2. * M_PI) * conf * kjy;
    } // for direction

    // re-calculate disparity as d=\Gamma^{-1}\Phi of the (low frequency) wavelet scales that we used up to now
    double gamma_det = gamma_x_x * gamma_y_y - sqr(gamma_x_y);
    m_disparity[1] = (gamma_y_y * phi_x - gamma_x_y * phi_y) / gamma_det;
    m_disparity[0] = (gamma_x_x * phi_y - gamma_x_y * phi_x) / gamma_det;

  } // for level
}

/**
 * Computes the similarity of the given Gabor jets by first estimating the disparity vector \f$ \vec d \f$
 * and afterwards using this disparity to correct the Gabor phase difference:
 * \f[ \sum_j a_j\,a_j'\,\cos(\phi_j - \phi_j' - \vec d^T \vec k_j) \f]
 * @param jet1 One of the two Gabor jets to compare
 * @param jet2 One of the two Gabor jets to compare
 * @return The similarity of jet1 and jet2, a value between -1 and 1
 */
double bob::machine::DisparitySimilarity::similarity(const blitz::Array<double,2>& jet1, const blitz::Array<double,2>& jet2) const{
  // compute confidence vectors
  compute_confidences(jet1, jet2);

  // now, compute the disparity
  compute_disparity();

  // finally, compute the similarity using the estimated disparity
  double sum = 0.;
  for (int j = m_confidences.size(); j--;){
    sum += m_confidences[j] * cos(m_phase_differences[j] - m_disparity[0] * m_kernel_frequencies[j][0] - m_disparity[1] * m_kernel_frequencies[j][1]);
  }

  return sum;
}

/**
 * Computes the similarity of the given Gabor jets by first estimating the disparity vector \f$ \vec d \f$
 * and afterwards using this disparity to correct the Gabor phase difference:
 * \f[ \frac1J \sum_j \cos(\phi_j - \phi_j' - \vec d^T \vec k_j) \f]
 * @param jet1 One of the two Gabor jets to compare
 * @param jet2 One of the two Gabor jets to compare
 * @return The similarity of jet1 and jet2, a value between -1 and 1
 */
double bob::machine::DisparityCorrectedPhaseDifference::similarity(const blitz::Array<double,2>& jet1, const blitz::Array<double,2>& jet2) const{
  // compute confidence vectors
  compute_confidences(jet1, jet2);

  // now, compute the disparity
  compute_disparity();

  // finally, compute the similarity using the estimated disparity
  double sum = 0.;
  for (int j = m_phase_differences.size(); j--;){
    sum += cos(m_phase_differences[j] - m_disparity[0] * m_kernel_frequencies[j][0] - m_disparity[1] * m_kernel_frequencies[j][1]);
  }

  return sum / jet1.shape()[1];
}


/**
 * Computes the similarity of the given Gabor jets by first estimating the disparity vector \f$ \vec d \f$
 * and afterwards using this disparity to correct the Gabor phase difference:
 * \f[ \frac1{2J} \sum_j \left[ \cos(\phi_j - \phi_j' - \vec d^T \vec k_j) + \frac{|a_j - a_j'|}{a_j + a_j'} \right]\f]
 * @param jet1 One of the two Gabor jets to compare
 * @param jet2 One of the two Gabor jets to compare
 * @return The similarity of jet1 and jet2, a value between -1 and 1
 */
double bob::machine::DisparityCorrectedPhaseDifferencePlusCanberra::similarity(const blitz::Array<double,2>& jet1, const blitz::Array<double,2>& jet2) const{
  // compute confidence vectors
  compute_confidences(jet1, jet2);

  // now, compute the disparity
  compute_disparity();

  // finally, compute the similarity using the estimated disparity
  double sum = 0.;
  for (int j = m_phase_differences.size(); j--;){
    // add disparity term
    sum += cos(m_phase_differences[j] - m_disparity[0] * m_kernel_frequencies[j][0] - m_disparity[1] * m_kernel_frequencies[j][1]);
    // add Canberra term
    sum += 1. - std::abs(jet1(0,j) - jet2(0,j)) / (jet1(0,j) + jet2(0,j));
  }

  return sum / (2. * jet1.shape()[1]);
}
