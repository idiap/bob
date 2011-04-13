/**
 * @file src/cxx/ip/ip/GaborFrequency.h
 * @author <a href="mailto:Laurent.El-Shafey@idiap.ch">Laurent El Shafey</a> 
 *
 * @brief This file provides a class to filter a 2D image/array with a Gabor
 * filter in the frequency domain.
 */

#ifndef TORCH5SPRO_IP_GABOR_FREQUENCY_H
#define TORCH5SPRO_IP_GABOR_FREQUENCY_H

#include "ip/Exception.h"

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
          // const enum ip::Gabor::NormOption norm_opt=ip::Gabor::SpatialFactor);

        /**
         * @brief Destructor
         */
        virtual ~GaborFrequency();

        /**
         * @brief Process a 2D blitz Array/Image by applying the Gabor filter
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
        // inline enum ip::Gabor::NormOption getNormOption() const
          // { return m_norm_opt; }

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
        // inline void setNormOption(const enum ip::Gabor::NormOption norm_opt)
          // { m_norm_opt = norm_opt; computeFilter(); }

      private:
        /**
         * @brief Generate the frequency Gabor filter. This is a Gaussian in
         *   the frequency domain
         */
        void computeFilter();

        /**
         * @brief Initialize the two working arrays
         */
        void initWorkArrays();

        /** 
          * @brief Compute an ellipsoid envelope which contains pf percent of 
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
//        enum ip::Gabor::NormOption m_norm_opt;
    };
}}

#endif /* TORCH5SPRO_IP_GABOR_FREQUENCY_H */
