/**
 * @file cxx/ip/ip/GaborBankSpatial.h
 * @date Wed Apr 13 20:45:32 2011 +0200
 * @author Laurent El Shafey <Laurent.El-Shafey@idiap.ch>
 *
 * @brief This file provides a class to filter a 2D image/array with a Gabor
 * filter bank in the spatial domain.
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

#ifndef BOB5SPRO_IP_GABOR_BANK_SPATIAL_H
#define BOB5SPRO_IP_GABOR_BANK_SPATIAL_H 

#include "core/cast.h"
#include "ip/Exception.h"
#include "ip/GaborSpatial.h"
#include <boost/shared_ptr.hpp>
#include <vector>

namespace bob {
/**
 * \ingroup libip_api
 * @{
 *
 */
  namespace ip {

    /**
     * @brief This class can be used to perform Gabor filtering.
     */
    class GaborBankSpatial
    {
      public:

        /**
         * @brief Constructor: generates the Gabor filter
         */
        GaborBankSpatial(const int n_orient=8, const int n_freq=5, 
          const double fmax=0.25, const bool orientation_full=false,
          const double k=1.414, const double p=0.5, 
          // Gabor Spatial filter options
          const double gamma=1., const double eta=1.,
          const int spatial_size=35, const bool cancel_dc=false, 
          const enum ip::Gabor::NormOption norm_opt=ip::Gabor::SpatialFactor,
          // const enum sp::Convolution::SizeOption size_opt=sp::Convolution::Same,
          const enum sp::Convolution::BorderOption border_opt=sp::Convolution::Mirror);

        /**
         * @brief Destructor
         */
        virtual ~GaborBankSpatial();

        /**
         * @brief Process a 2D blitz Array/Image by applying the Gabor filter bank
         */
        void operator()( const blitz::Array<std::complex<double>,2>& src,
            blitz::Array<std::complex<double>,3>& dst);

        /**
          * @brief Accessor functions
          */
        inline int getNOrient() const { return m_n_orient; }
        inline int getNFreq() const { return m_n_freq; }
        inline double getFmax() const { return m_fmax; }
        inline bool getOrientationFull() const { return m_orientation_full; }
        inline double getK() const { return m_k; }
        inline double getP() const { return m_p; }
        inline double getGamma() const { return m_gamma; }
        inline double getEta() const { return m_eta; }
        inline int getSpatialSize() const { return m_spatial_size; }
        inline bool getCancelDc() const { return m_cancel_dc; }
        inline enum ip::Gabor::NormOption getNormOption() const
          { return m_norm_opt; }
        inline enum sp::Convolution::BorderOption getBorderOption() const
          { return m_border_opt; }

        /**
          * @brief Mutator functions
          */
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
        inline void setGamma(const double gamma) 
          { m_gamma = gamma; computeFilters(); }
        inline void setEta(const double eta) 
          { m_eta = eta; computeFilters(); }
        inline void setSpatialSize(const int spatial_size) 
          { m_spatial_size = spatial_size; computeFilters(); }
        inline void setCancelDc(const bool cancel_dc) 
          { m_cancel_dc = cancel_dc; computeFilters(); }
        inline void setNormOption(const enum ip::Gabor::NormOption norm_opt)
          { m_norm_opt = norm_opt; computeFilters(); }
        inline void setBorderOption( const enum sp::Convolution::BorderOption 
            border_opt) 
          { m_border_opt = border_opt; }

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
        int m_n_orient;
        int m_n_freq;
        double m_fmax;
        bool m_orientation_full;
        double m_k;
        double m_p;
        double m_gamma;
        double m_eta;
        int m_spatial_size;
        bool m_cancel_dc;
        enum ip::Gabor::NormOption m_norm_opt;
        // enum sp::Convolution::SizeOption m_size_opt;
        enum sp::Convolution::BorderOption m_border_opt;

        std::vector<boost::shared_ptr<bob::ip::GaborSpatial> > m_filters;
        blitz::Array<double,1> m_freqs;
        blitz::Array<double,1> m_orients;
    };
}}

#endif /* BOB5SPRO_IP_GABOR_BANK_SPATIAL_H */
