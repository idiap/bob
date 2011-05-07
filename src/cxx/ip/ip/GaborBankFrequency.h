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
          const double p=0.5, const bool optimal_gamma_eta=false,
          // Gabor Frequency filter options
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

        /**
          * @brief Accessor functions
          */
        inline int getHeight() const { return m_height; }
        inline int getWidth() const { return m_width; }
        inline int getNOrient() const { return m_n_orient; }
        inline int getNFreq() const { return m_n_freq; }
        inline double getFmax() const { return m_fmax; }
        inline bool getOrientationFull() const { return m_orientation_full; }
        inline double getK() const { return m_k; }
        inline double getP() const { return m_p; }
        inline bool getOptimalGammaEta() const { return m_optimal_gamma_eta; }
        inline double getGamma() const { return m_gamma; }
        inline double getEta() const { return m_eta; }
        inline double getPf() const { return m_pf; }
        inline bool getCancelDc() const { return m_cancel_dc; }
        inline bool getUseEnvelope() const
          { return m_use_envelope; }
        inline bool getOutputInFrequency() const
          { return m_output_in_frequency; }

        /**
          * @brief Mutator functions
          */
        inline void setHeight(const int height) 
          { m_height = height; computeFilters(); }
        inline void setWidth(const int width) 
          { m_width = width; computeFilters(); }
        inline void setNOrient(const int n_orient)
          { m_n_orient = n_orient; computeFilters(); }
        inline void setNFreq(const int n_freq)
          { m_n_freq = n_freq; computeFilters(); }
        inline void setFmax(const double fmax) 
          { m_fmax = fmax; computeFilters(); }
        inline void setOrientationFull(const bool orientation_full) 
          { m_orientation_full = orientation_full; computeFilters(); }
        inline void setK(const double k) 
          { m_k = k; computeFilters(); }
        inline void setP(const double p) 
          { m_p = p; computeFilters(); }
        inline void setOptimalGammaEta(const bool opt)
          { m_optimal_gamma_eta = opt; computeFilters(); }
        inline void setGamma(const double gamma) 
          { m_gamma = gamma; computeFilters(); }
        inline void setEta(const double eta) 
          { m_eta = eta; computeFilters(); }
        inline void setPf(const double pf) 
          { m_pf = pf; computeFilters(); }
        inline void setCancelDc(const bool cancel_dc) 
          { m_cancel_dc = cancel_dc; computeFilters(); }
        inline void setUseEnvelope(const bool use_envelope)
          { m_use_envelope = use_envelope; computeFilters(); }
        inline void setOutputInFrequency( const bool output_in_frequency)
          { m_output_in_frequency = output_in_frequency; }



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

        /**
         * @brief Compute and set "optimal" gamma and eta from m_n_orient, 
         *   m_orientation_full, m_k, m_p as described in:
         * "Rotation-invariant and scale-invariant Gabor features for texture
         *  image retrieval", by J. Han and K.K. Ma 
         * in Image and Vision Computing 25 (2007), 1474-1481
         */
        void computeOptimalGammaEta();

        // Attributes
        int m_height;
        int m_width;
        int m_n_orient;
        int m_n_freq;
        double m_fmax;
        bool m_orientation_full;
        double m_k;
        double m_p;
        bool m_optimal_gamma_eta;
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
