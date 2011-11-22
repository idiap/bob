/**
 * @file cxx/ip/ip/GaborFrequency.h
 * @date Wed Apr 13 20:12:03 2011 +0200
 * @author Laurent El Shafey <Laurent.El-Shafey@idiap.ch>
 *
 * @brief This file provides a class to filter a 2D image/array with a Gabor
 * filter in the frequency domain.
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

#ifndef TORCH5SPRO_IP_GABOR_FREQUENCY_H
#define TORCH5SPRO_IP_GABOR_FREQUENCY_H

#include <boost/shared_ptr.hpp>
#include "ip/Exception.h"
#include "sp/FFT2D.h"
#include <complex>

namespace Torch {
/**
 * \ingroup libip_api
 * @{
 *
 */
  namespace ip {

    /**
     * @brief Enumerations of the possible options
     */
/*    namespace Gabor {
      enum NormOption {
        NoNorm,
        SpatialFactor,
        ZeroMeanUnitVar
      };
    }
*/
    /**
     * @brief This class can be used to perform Gabor filtering in the 
     * frequency domain. Please refer to the following article for more
     * information:
     *   "Invariance properties of Gabor filter-based features-overview and 
     *   applications ", J.K. Kamarainen, V. Kyrki, H. Kalviainen, 
     *   in IEEE Transactions on Image Procesing, vol. 15, Issue 5, 
     *   pp. 1088-1099 
     */
    class GaborFrequency
    {
      public:

        /**
         * @brief Constructor: generates the Gabor filter
         */
        GaborFrequency(const int height, const int width, const double f=0.25,
          const double theta=0., const double gamma=1., const double eta=1., 
          const double pf=0.99, const bool cancel_dc=false, 
          const bool use_envelope=false, const bool output_in_frequency=false);

      
        /**
         * @brief Copy constructor
         */
        GaborFrequency(const GaborFrequency& copy);

        /**
         * @brief Destructor
         */
        virtual ~GaborFrequency();

        /**
         * @brief Processes a 2D blitz Array/Image by applying the Gabor filter
         */
        void operator()( const blitz::Array<std::complex<double>,2>& src,
            blitz::Array<std::complex<double>,2>& dst);
  
        /**
          * @brief Accessor functions
          */
        inline int getHeight() const { return m_height; }
        inline int getWidth() const { return m_width; }
        inline double getF() const { return m_f; }
        inline double getTheta() const { return m_theta; }
        inline double getGamma() const { return m_gamma; }
        inline double getEta() const { return m_eta; }
        inline double getPf() const { return m_pf; }
        inline bool getCancelDc() const { return m_cancel_dc; }
        inline bool getUseEnvelope() const { return m_use_envelope; }

        /**
          * @brief Returns the frequency filter, the zero frequency being at 
          * the corner
          */
        inline const blitz::Array<std::complex<double>, 2>& getKernel() const
          { return m_kernel; }
        /**
          * @brief Returns the frequency filter, the zero frequency being at 
          * the center
          */
        inline const blitz::Array<std::complex<double>, 2>& getKernelShifted()
          const { return m_kernel_shifted; }
        /**
          * @brief Returns the 'envelope' frequency filter, which contains pf
          *   percent of the frequency filter, the zero frequency being at 
          *   the center, OR the full center if no envelope is used for 
          *   filtering.
          */
        inline const blitz::Array<std::complex<double>, 2> 
          getKernelEnvelope() const 
        { 
          if( !m_use_envelope)
            return m_kernel_shifted;
          else
           return m_kernel_shifted( blitz::Range(m_env_y_min,m_env_y_max),
              blitz::Range(m_env_x_min,m_env_x_max) );
        }

        /**
          * @brief Mutator functions
          */
        inline void setHeight(const int height) 
          { m_height = height; computeFilter(); initWorkArrays(); }
        inline void setWidth(const int width) 
          { m_width = width; computeFilter(); initWorkArrays(); }
        inline void setF(const double f) 
          { m_f = f; computeFilter(); }
        inline void setTheta(const double theta) 
          { m_theta = theta; computeFilter(); }
        inline void setGamma(const double gamma) 
          { m_gamma = gamma; computeFilter(); }
        inline void setEta(const double eta) 
          { m_eta = eta; computeFilter(); }
        inline void setPf(const double pf) 
          { m_pf = pf; computeFilter(); }
        inline void setCancelDc(const bool cancel_dc) 
          { m_cancel_dc = cancel_dc; computeFilter(); }
        inline void setUseEnvelope(const bool use_envelope) 
          { m_use_envelope = use_envelope; computeFilter(); }

      private:
        /**
         * @brief Generates the frequency Gabor filter. This is a Gaussian in
         *   the frequency domain
         */
        void computeFilter();

        /**
         * @brief Initializes the two working arrays
         */
        void initWorkArrays();

        /** 
          * @brief Computes an ellipsoid envelope which contains pf percent of 
          *   the energy of the whole frequency filter
          */
        void computeEnvelope();

        // Attributes
        int m_height;
        int m_width;
        double m_f;
        double m_theta;
        double m_gamma;
        double m_eta;
        double m_pf;
        bool m_cancel_dc;
        bool m_use_envelope;
        bool m_output_in_frequency;

        blitz::Array<std::complex<double>, 2> m_kernel_shifted;
        blitz::Array<std::complex<double>, 2> m_kernel;
        int m_env_height;
        int m_env_width;
        int m_env_y_min;
        int m_env_y_max;
        int m_env_x_min;
        int m_env_x_max;
        int m_env_y_offset;
        int m_env_x_offset;
        // Intermediate working arrays
        blitz::Array<std::complex<double>, 2> m_work1;
        blitz::Array<std::complex<double>, 2> m_work2;
        boost::shared_ptr<Torch::sp::FFT2D> m_fft;
        boost::shared_ptr<Torch::sp::IFFT2D> m_ifft;
    };
}}

#endif /* TORCH5SPRO_IP_GABOR_FREQUENCY_H */
