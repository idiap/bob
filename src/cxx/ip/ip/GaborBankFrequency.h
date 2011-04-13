/**
 * @file src/cxx/ip/ip/GaborBankFrequency.h
 * @author <a href="mailto:Laurent.El-Shafey@idiap.ch">Laurent El Shafey</a> 
 *
 * @brief This file provides a class to filter a 2D image/array with a Gabor
 * filter bank in the frequency domain.
 */

#ifndef TORCH5SPRO_IP_GABOR_BANK_FREQUENCY_H
#define TORCH5SPRO_IP_GABOR_BANK_FREQUENCY_H

#include "core/cast.h"
#include "ip/Exception.h"
#include "ip/GaborFrequency.h"
#include <boost/shared_ptr.hpp>
#include <vector>

namespace Torch {
/**
 * \ingroup libip_api
 * @{
 *
 */
  namespace ip {

    /**
     * @brief This class can be used to perform Gabor filtering.
     */
    class GaborBankFrequency
    {
      public:

        /**
         * @brief Constructor: generates the Gabor filter
         */
        GaborBankFrequency(const int height, const int width, 
          const int n_orient=8, const int n_freq=5, const double fmax=0.25, 
          const bool orientation_full=false, const double k=1.414, 
          const double p=0.5, 
          // Gabor Spatial filter options
          const double gamma=1., const double eta=1.,
          const double pf=0.99, const bool cancel_dc=false, 
          const bool use_envelope=false, const bool output_in_frequency=false);

        /**
         * @brief Destructor
         */
        virtual ~GaborBankFrequency();

        /**
         * @brief Process a 2D blitz Array/Image by applying the Gabor filter bank
         */
        void operator()( const blitz::Array<std::complex<double>,2>& src,
            blitz::Array<std::complex<double>,3>& dst);

      private:
        /**
         * @brief Generate the frequencies
         */
        void computeFreqs();

        /**
         * @brief Generate the orientations
         */
        void computeOrients();

        /**
         * @brief Generate the spatial Gabor filter
         */
        void computeFilters();

        // Attributes
        int m_height;
        int m_width;
        int m_n_orient;
        int m_n_freq;
        double m_fmax;
        double m_orientation_full;
        double m_k;
        double m_p;
        double m_gamma;
        double m_eta;
        double m_pf;
        bool m_cancel_dc;
        bool m_use_envelope;
        bool m_output_in_frequency;

        std::vector<boost::shared_ptr<Torch::ip::GaborFrequency> > m_filters;
        blitz::Array<double,1> m_freqs;
        blitz::Array<double,1> m_orients;
    };
}}

#endif /* TORCH5SPRO_IP_GABOR_BANK_FREQUENCY_H */
