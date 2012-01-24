/**
 * @file cxx/measure/measure/GaborJetSimilarity.h
 * @date
 * @author
 *
 * @brief A set of methods that evaluates error from score sets
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

#ifndef BOB_MEASURE_GABOR_JET_SIMILARITY_H
#define BOB_MEASURE_GABOR_JET_SIMILARITY_H

#include <core/array_assert.h>
#include <blitz/array.h>
#include <numeric>
#include <ip/GaborWaveletTransform.h>

namespace bob { namespace measure {

  //! base class for Gabor jet similarity functions
  class GaborJetSimilarity{
    protected:
      GaborJetSimilarity(){}
      virtual ~GaborJetSimilarity(){}

    public:
      //! The similarity between two Gabor jets
      virtual double similarity(const blitz::Array<double,2>& jet1, const blitz::Array<double,2>& jet2) const = 0;

  };

  //! The default Gabor jet similarity function, which is the normalized scalar product,
  //! also known as the cosine measure
  class ScalarProductSimilarity : GaborJetSimilarity{
    public:
      ScalarProductSimilarity() : GaborJetSimilarity(){}
      virtual ~ScalarProductSimilarity(){}

      virtual double similarity(const blitz::Array<double,2>& jet1, const blitz::Array<double,2>& jet2) const;
  };

  //! The default Gabor jet similarity function, which is the normalized scalar product,
  //! also known as the cosine measure
  class CanberraSimilarity : GaborJetSimilarity{
    public:
      CanberraSimilarity() : GaborJetSimilarity(){}
      virtual ~CanberraSimilarity(){}

      virtual double similarity(const blitz::Array<double,2>& jet1, const blitz::Array<double,2>& jet2) const;
  };

  //! This function computes the disparity similarity function by estimating a disparity vector from two jets
  //! and using the disparity for phase difference correction
  class DisparitySimilarity : GaborJetSimilarity{
    public:
      DisparitySimilarity(const bob::ip::GaborWaveletTransform& gwt = bob::ip::GaborWaveletTransform());
      virtual ~DisparitySimilarity(){}

      //! computes the similarity (including the estimation of the disparity vector)
      virtual double similarity(const blitz::Array<double,2>& jet1, const blitz::Array<double,2>& jet2) const;
      //! returns the disparity vector estimated during the last call of similarity(...)
      const std::pair<double,double>& disparity() const {return m_disparity;}

    private:
      const std::vector<std::pair<double,double> > m_kernel_frequencies;
      const int m_number_of_scales;
      const int m_number_of_directions;
      std::vector<double> m_wavelet_extends;

      mutable std::pair<double,double> m_disparity;
      mutable std::vector<double> m_confidences;
      mutable std::vector<double> m_phase_differences;
};
} }

#endif // BOB_MEASURE_GABOR_JET_SIMILARITY_H
