/**
 * @date 2012-03-05
 * @author Manuel Guenther <Manuel.Guenther@idiap.ch>
 *
 * @brief Similarity functions of two Gabor jets.
 *
 * Copyright (C) Idiap Research Institute, Martigny, Switzerland
 */

#ifndef BOB_MACHINE_GABOR_JET_SIMILARITY_H
#define BOB_MACHINE_GABOR_JET_SIMILARITY_H

#include <bob/core/assert.h>
#include <blitz/array.h>
#include <numeric>
#include <bob/ip/GaborWaveletTransform.h>

namespace bob { namespace machine {
  /**
   * @ingroup MACHINE
   * @{
   */

  //! Class to compute Gabor jet similarities
  class GaborJetSimilarity{
    public:

      //! This enum defines different types of Gabor jet similarity functions.
      //! The first functions are based on absolute values of Gabor jets,
      //! while the latter also use the Gabor phases
      typedef enum {
        SCALAR_PRODUCT = 1,
        CANBERRA = 3,
        DISPARITY = 16,
        PHASE_DIFF = 22,
        PHASE_DIFF_PLUS_CANBERRA = 30
      }
      SimilarityType;

      //! Constructor for the Gabor jet similarity
      GaborJetSimilarity(SimilarityType type, const bob::ip::GaborWaveletTransform& gwt = bob::ip::GaborWaveletTransform());

      //! The similarity between two Gabor jets, including absolute values and phases
      double operator()(const blitz::Array<double,2>& jet1, const blitz::Array<double,2>& jet2) const;

      //! The similarity between two Gabor jets, including absolute values only
      double operator()(const blitz::Array<double,1>& jet1, const blitz::Array<double,1>& jet2) const;

      //! returns the disparity vector estimated during the last call of similarity; only valid for disparity types
      blitz::TinyVector<double,2> disparity() const {return m_disparity;}

      //! \brief saves the parameters of this Gabor jet similarity to file
      void save(bob::io::HDF5File& file) const;

      //! \brief reads the parameters of this Gabor jet similarity from file
      void load(bob::io::HDF5File& file);

    private:
      // members for all similarity functions
      SimilarityType m_type;

      // members required by disparity functions
      bob::ip::GaborWaveletTransform m_gwt;

      // initializes the internal memory to be used for disparity-like Gabor jet similarities
      void init();
      // computes confidences from the given Gabor jets
      void compute_confidences(const blitz::Array<double,2>& jet1, const blitz::Array<double,2>& jet2) const;
      // computes the disparity using the m_confidences and m_phase_differences values
      void compute_disparity() const;

      mutable blitz::TinyVector<double,2> m_disparity;

      mutable std::vector<double> m_confidences;
      mutable std::vector<double> m_phase_differences;
      std::vector<double> m_wavelet_extends;

  }; // class GaborJetSimilarity

  /**
   * @}
   */
} } // namespaces

#endif // BOB_MACHINE_GABOR_JET_SIMILARITY_H
