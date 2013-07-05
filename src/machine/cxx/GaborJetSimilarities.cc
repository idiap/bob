/**
 * @file machine/cxx/GaborJetSimilarities.cc
 * @date 2012-03-05
 * @author Manuel Guenther <Manuel.Guenther@idiap.ch>
 *
 * @brief Implements the Gabor jet similarity functions.
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

#include "bob/machine/GaborJetSimilarities.h"

bob::machine::GaborJetSimilarity::GaborJetSimilarity(bob::machine::GaborJetSimilarity::SimilarityType type, const bob::ip::GaborWaveletTransform& gwt)
:
  m_type(type),
  m_gwt(gwt)
{
  // initialize, when required
  if (m_type >= DISPARITY){
    init();
  }
}

static double sqr(double x){return x*x;}

void bob::machine::GaborJetSimilarity::init(){
  m_disparity = 0.;
  m_confidences.resize(m_gwt.numberOfKernels());
  std::fill(m_confidences.begin(), m_confidences.end(), 0.);
  m_phase_differences.resize(m_gwt.numberOfKernels());
  std::fill(m_phase_differences.begin(), m_phase_differences.end(), 0.);

  // used for disparity-like similarity functions only...
  m_wavelet_extends.reserve(m_gwt.numberOfScales());
  for (unsigned level = 0; level < m_gwt.numberOfScales(); ++level){
    blitz::TinyVector<double,2> k = m_gwt.kernelFrequencies()[level * m_gwt.numberOfDirections()];
    double k_abs = sqrt(sqr(k[0]) + sqr(k[1]));
    m_wavelet_extends.push_back(M_PI / k_abs);
  }
}


double bob::machine::GaborJetSimilarity::operator()(const blitz::Array<double,1>& jet1, const blitz::Array<double,1>& jet2) const{
  bob::core::array::assertCZeroBaseContiguous(jet1);
  bob::core::array::assertCZeroBaseContiguous(jet2);
  bob::core::array::assertSameShape(jet1,jet2);

  switch (m_type){
    case SCALAR_PRODUCT:
      // normalized scalar product
      return std::inner_product(jet1.begin(), jet1.end(), jet2.begin(), 0.);
    case CANBERRA:{
      // Canberra similarity
      double sim = 0.;
      unsigned size = jet1.shape()[0];
      for (unsigned j = size; j--;){
        sim += 1. - std::abs(jet1(j) - jet2(j)) / (jet1(j) + jet2(j));
      }
      return sim / size;
    }
    default:
      throw std::runtime_error("Disparity similarity (and its derivatives) need Gabor jets including phases");
  }
}


double bob::machine::GaborJetSimilarity::operator()(const blitz::Array<double,2>& jet1, const blitz::Array<double,2>& jet2) const{
  if (m_type == SCALAR_PRODUCT || m_type == CANBERRA){
    // call the function without phases
    return operator()(jet1(0,blitz::Range::all()), jet2(0,blitz::Range::all()));
  }

  // Here, only the disparity based similarity functions are executed
  bob::core::array::assertCZeroBaseContiguous(jet1);
  bob::core::array::assertCZeroBaseContiguous(jet2);
  bob::core::array::assertSameShape(jet1,jet2);

  // compute confidence vectors
  compute_confidences(jet1, jet2);

  // now, compute the disparity
  compute_disparity();

  const std::vector<blitz::TinyVector<double,2> >& kernels = m_gwt.kernelFrequencies();

  switch (m_type){
    case DISPARITY:{
      // compute the similarity using the estimated disparity
      double sum = 0.;
      for (int j = m_confidences.size(); j--;){
        sum += m_confidences[j] * cos(m_phase_differences[j] - m_disparity[0] * kernels[j][0] - m_disparity[1] * kernels[j][1]);
      }
      return sum;
    } // DISPARITY

    case PHASE_DIFF:{
      // compute the similarity using the estimated disparity
      double sum = 0.;
      for (int j = m_phase_differences.size(); j--;){
        sum += cos(m_phase_differences[j] - m_disparity[0] * kernels[j][0] - m_disparity[1] * kernels[j][1]);
      }
      return sum / jet1.shape()[1];
    } // PHASE_DIFF

    case PHASE_DIFF_PLUS_CANBERRA:{
      // compute the similarity using the estimated disparity
      double sum = 0.;
      for (int j = m_phase_differences.size(); j--;){
        // add disparity term
        sum += cos(m_phase_differences[j] - m_disparity[0] * kernels[j][0] - m_disparity[1] * kernels[j][1]);
        // add Canberra term
        sum += 1. - std::abs(jet1(0,j) - jet2(0,j)) / (jet1(0,j) + jet2(0,j));
      }
      return sum / (2. * jet1.shape()[1]);
    }

    default:
      // this should never happen
      throw std::runtime_error("This should not have happened. Please check the implementation of the similarity() functions.");
  }
}


/////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////  Disparity estimation  /////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////
static double adjustPhase(double phase){
  return phase - (2.*M_PI)*round(phase / (2.*M_PI));
}

void bob::machine::GaborJetSimilarity::compute_confidences(const blitz::Array<double,2>& jet1, const blitz::Array<double,2>& jet2) const{
  // first, fill confidence and phase difference vectors
  for (int j = m_confidences.size(); j--;){
    m_confidences[j] = jet1(0,j) * jet2(0,j);
    m_phase_differences[j] = adjustPhase(jet1(1,j) - jet2(1,j));
  }
}

void bob::machine::GaborJetSimilarity::compute_disparity() const{
  // approximate the disparity from the phase differences
  double gamma_x_x = 0., gamma_x_y = 0., gamma_y_y = 0., phi_x = 0., phi_y = 0.;
  // initialize the disparity with 0
  m_disparity = 0.;

  const std::vector<blitz::TinyVector<double,2> >& kernels = m_gwt.kernelFrequencies();
  // iterate backwards through the vector to start with the lowest frequency wavelets
  for (int j = m_confidences.size()-1, level = m_gwt.numberOfScales()-1; level >= 0; --level){
    for (int direction = m_gwt.numberOfDirections()-1; direction >= 0; --direction, --j){
      double
          kjx = kernels[j][1],
          kjy = kernels[j][0],
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


void bob::machine::GaborJetSimilarity::save(bob::io::HDF5File& file) const{

  file.set("Type", m_type);
#undef TYPE_TO_STRING
  if (m_type >= DISPARITY){
    file.createGroup("GaborWaveletTransform");
    file.cd("GaborWaveletTransform");
    m_gwt.save(file);
    file.cd("..");
  }
}


void bob::machine::GaborJetSimilarity::load(bob::io::HDF5File& file){
  // read value
  m_type = (SimilarityType)file.read<int>("Type");

  if (m_type >= DISPARITY){
    file.cd("GaborWaveletTransform");
    m_gwt.load(file);
    file.cd("..");

    init();
  }
}
