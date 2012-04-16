/**
 * @file cxx/machine/machine/GaborJetSimilarities.h
 * @date 2012-03-05
 * @author Manuel Guenther <Manuel.Guenther@idiap.ch>
 *
 * @brief Similarity functions of two Gabor jets.
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

#ifndef BOB_MACHINE_GABOR_JET_SIMILARITY_H
#define BOB_MACHINE_GABOR_JET_SIMILARITY_H

#include <core/array_assert.h>
#include <blitz/array.h>
#include <numeric>
#include <ip/GaborWaveletTransform.h>

namespace bob { namespace machine {

  //! base class for Gabor jet similarity functions
  class GaborJetSimilarity{
    public:
      GaborJetSimilarity(){}

      //! The similarity between two Gabor jets, including absolute values and phases
      virtual double similarity(const blitz::Array<double,2>& jet1, const blitz::Array<double,2>& jet2) const = 0;

      //! The similarity between two Gabor jets, including absolute values only
      virtual double similarity(const blitz::Array<double,1>& jet1, const blitz::Array<double,1>& jet2) const = 0;

  };

  //! \brief The default Gabor jet similarity function, which is the normalized scalar product,
  //! also known as the cosine measure
  class ScalarProductSimilarity : public GaborJetSimilarity{
    public:
      ScalarProductSimilarity() : GaborJetSimilarity(){}

      //! computes similarity between Gabor jets with absolute and phase values
      virtual double similarity(const blitz::Array<double,2>& jet1, const blitz::Array<double,2>& jet2) const;

      //! computes similarity between Gabor jets with absolute values only
      virtual double similarity(const blitz::Array<double,1>& jet1, const blitz::Array<double,1>& jet2) const;
};

  //! The Canberra similarity measure which most often performs better than the cosine
  class CanberraSimilarity : public GaborJetSimilarity{
    public:
      CanberraSimilarity() : GaborJetSimilarity(){}

      //! computes similarity between Gabor jets with absolute and phase values
      virtual double similarity(const blitz::Array<double,2>& jet1, const blitz::Array<double,2>& jet2) const;

      //! computes similarity between Gabor jets with absolute values only
      virtual double similarity(const blitz::Array<double,1>& jet1, const blitz::Array<double,1>& jet2) const;
  };

  //! \brief This function computes the disparity similarity function by estimating a disparity vector from two jets
  //! and using the disparity for phase difference correction
  class DisparitySimilarity : public GaborJetSimilarity{
    public:
      DisparitySimilarity(const bob::ip::GaborWaveletTransform& gwt = bob::ip::GaborWaveletTransform());

      //! computes the similarity (including the estimation of the disparity vector)
      virtual double similarity(const blitz::Array<double,2>& jet1, const blitz::Array<double,2>& jet2) const;

      //! the similarity between Gabor jets without phases is not supported by this class (and its derivations)
      virtual double similarity(const blitz::Array<double,1>& jet1, const blitz::Array<double,1>& jet2) const{
        throw bob::core::NotImplementedError("Disparity similarity (and its derivatives) need Gabor jets including phases");
      }

      //! returns the disparity vector estimated during the last call of similarity(...)
      const blitz::TinyVector<double,2>& disparity() const {return m_disparity;}

    protected:
      virtual void compute_confidences(const blitz::Array<double,2>& jet1, const blitz::Array<double,2>& jet2) const;
      //! computes the disparity using the m_confidences and m_phase_differences values
      virtual void compute_disparity() const;

      mutable blitz::TinyVector<double,2> m_disparity;
      mutable std::vector<double> m_confidences;
      mutable std::vector<double> m_phase_differences;
      const std::vector<blitz::TinyVector<double,2> > m_kernel_frequencies;

    private:
      const int m_number_of_scales;
      const int m_number_of_directions;
      std::vector<double> m_wavelet_extends;

  };

  //! \brief This class computes the similarity using disparity corrected phase differences
  //! (without accounting for the absolute values)
  class DisparityCorrectedPhaseDifference : public DisparitySimilarity{
    public:
      DisparityCorrectedPhaseDifference(const bob::ip::GaborWaveletTransform& gwt = bob::ip::GaborWaveletTransform())
      :DisparitySimilarity(gwt){}

      //! computes the similarity using only the phases
      virtual double similarity(const blitz::Array<double,2>& jet1, const blitz::Array<double,2>& jet2) const;
  };


  //! \brief This class computes the similarity using disparity corrected phase differences
  //! for the phases and Canberra similarity for the absolute values.
  class DisparityCorrectedPhaseDifferencePlusCanberra : public DisparitySimilarity{
    public:
      DisparityCorrectedPhaseDifferencePlusCanberra(const bob::ip::GaborWaveletTransform& gwt = bob::ip::GaborWaveletTransform())
      :DisparitySimilarity(gwt){}

      //! computes the similarity using only the phases
      virtual double similarity(const blitz::Array<double,2>& jet1, const blitz::Array<double,2>& jet2) const;
  };

} }

#endif // BOB_MACHINE_GABOR_JET_SIMILARITY_H
