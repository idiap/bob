/**
 * @file cxx/ip/ip/GaborSpatial.h
 * @date Wed Apr 13 20:12:03 2011 +0200
 * @author Laurent El Shafey <Laurent.El-Shafey@idiap.ch>
 *
 * @brief This file provides a class to filter a 2D image/array with a Gabor
 * filter in the spatial domain.
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

#ifndef BOB5SPRO_IP_GABOR_SPATIAL_H
#define BOB5SPRO_IP_GABOR_SPATIAL_H

#include "ip/Exception.h"
#include "sp/convolution.h"

namespace bob {
/**
 * \ingroup libip_api
 * @{
 *
 */
  namespace ip {

    /**
     * @brief Enumerations of the possible options
     */
    namespace Gabor {
      enum NormOption {
        NoNorm,
        SpatialFactor,
        ZeroMeanUnitVar
      };
    }

    /**
     * @brief This class can be used to perform Gabor filtering in the
     * spatial domain. Please refer to the following article for more
     * information:
     *   "Invariance properties of Gabor filter-based features-overview and 
     *   applications ", J.K. Kamarainen, V. Kyrki, H. Kalviainen, 
     *   in IEEE Transactions on Image Procesing, vol. 15, Issue 5, 
     *   pp. 1088-1099 
     */
    // TODO: Deal with the SizeOption for convolution?
    class GaborSpatial
    {
      public:

        /**
         * @brief Constructor: generates the Gabor filter
         */
        GaborSpatial(const double f=0.25, const double theta=0., 
          const double gamma=1., const double eta=1., const int spatial_size=35,
          const bool cancel_dc=false, 
          const enum ip::Gabor::NormOption norm_opt=ip::Gabor::SpatialFactor,
          // const enum sp::Convolution::SizeOption size_opt=sp::Convolution::Same,
          const enum sp::Convolution::BorderOption border_opt=sp::Convolution::Mirror);

        /**
         * @brief Destructor
         */
        virtual ~GaborSpatial();

        /**
         * @brief Process a 2D blitz Array/Image by applying the Gabor filter
         */
        void operator()( const blitz::Array<std::complex<double>,2>& src,
            blitz::Array<std::complex<double>,2>& dst);

        /**
          * @brief Accessor functions
          */
        inline double getF() const { return m_f; }
        inline double getTheta() const { return m_theta; }
        inline double getGamma() const { return m_gamma; }
        inline double getEta() const { return m_eta; }
        inline int getSpatialSize() const { return m_spatial_size; }
        inline bool getCancelDc() const { return m_cancel_dc; }
        inline enum ip::Gabor::NormOption getNormOption() const
          { return m_norm_opt; }
        inline enum sp::Convolution::BorderOption getBorderOption() const
          { return m_border_opt; }
        inline const blitz::Array<std::complex<double>, 2>& getKernel() const
          { return m_kernel; }

        /**
          * @brief Mutator functions
          */
        inline void setF(const double f) 
          { m_f = f; computeFilter(); }
        inline void setTheta(const double theta) 
          { m_theta = theta; computeFilter(); }
        inline void setGamma(const double gamma) 
          { m_gamma = gamma; computeFilter(); }
        inline void setEta(const double eta) 
          { m_eta = eta; computeFilter(); }
        inline void setSpatialSize(const double spatial_size) 
          { m_spatial_size = spatial_size; computeFilter(); }
        inline void setCancelDc(const bool cancel_dc) 
          { m_cancel_dc = cancel_dc; computeFilter(); }
        inline void setNormOption(const enum ip::Gabor::NormOption norm_opt)
          { m_norm_opt = norm_opt; computeFilter(); }
        inline void setBorderOption( const enum sp::Convolution::BorderOption 
            border_opt) 
          { m_border_opt = border_opt; }

      private:
        /**
         * @brief Generate the spatial Gabor filter
         */
        void computeFilter();

        // Attributes
        blitz::Array<std::complex<double>, 2> m_kernel;
        double m_f;
        double m_theta;
        double m_gamma;
        double m_eta;
        int m_spatial_size;
        bool m_cancel_dc;
        enum ip::Gabor::NormOption m_norm_opt;
        //enum sp::Convolution::SizeOption m_size_opt;
        enum sp::Convolution::BorderOption m_border_opt;
    };
}}

#endif /* BOB5SPRO_IP_GABOR_SPATIAL_H */
