/**
 * @author Laurent El Shafey <Laurent.El-Shafey@idiap.ch>
 * @date Mon 29 Sep 2011
 */

#ifndef BOB_MACHINE_WIENERMACHINE_H
#define BOB_MACHINE_WIENERMACHINE_H

#include <blitz/array.h>
#include <complex>
#include "io/HDF5File.h"
#include "sp/FFT2D.h"

namespace bob { namespace machine {

  /**
   * A Wiener machine, which can be used to denoise a signal,
   * by comparing with a statistical model of the noiseless signal.
   * 
   * Computer Vision: Algorithms and Applications, Richard Szeliski
   * (Part 3.4.3)
   */
  class WienerMachine {

    public: //api

      /**
       * Default constructor. Builds an otherwise invalid 0 x 0 Wiener machine.
       * This is equivalent to construct a WienerMachine with two size_t
       * parameters set to 0, as in WienerMachine(0, 0).
       */
      WienerMachine ();

      /**
       * Constructor, builds a new Wiener machine. Wiener filter is initialized
       * with the given size, Ps being sets to the variance threshold.
       */
      WienerMachine (size_t height, size_t width, const double Pn, 
        const double variance_threshold=1e-8);

      /**
       * Builds a new machine with the given variance estimate Ps and noise
       * level Pn.
       */
      WienerMachine (const blitz::Array<double,2>& Ps, const double Pn, 
        const double variance_threshold=1e-8);

      /**
       * Builds a new machine with the given Wiener filter. 
       */
      WienerMachine(const blitz::Array<double,2>& W);

      /**
       * Copies another machine
       */
      WienerMachine (const WienerMachine& other);

      /**
       * Starts a new WienerMachine from an existing Configuration object.
       */
      WienerMachine (bob::io::HDF5File& config);

      /**
       * Just to virtualise the destructor
       */
      virtual ~WienerMachine();

      /**
       * Assigns from a different machine
       */
      WienerMachine& operator= (const WienerMachine& other);

      /**
       * Loads data from an existing configuration object. Resets the current
       * state.
       */
      void load (bob::io::HDF5File& config);

      /**
       * Saves an existing machine to a Configuration object.
       */
      void save (bob::io::HDF5File& config) const;

      /**
       * Forwards data through the machine.
       *
       * The input and output are NOT checked for compatibility each time. It
       * is your responsibility to do it.
       */
      void forward_ (const blitz::Array<double,2>& input,
          blitz::Array<double,2>& output) const;

      /**
       * Forwards data through the machine.
       *
       * The input and output are checked for compatibility each time the
       * forward method is applied.
       */
      void forward (const blitz::Array<double,2>& input,
          blitz::Array<double,2>& output) const;

      /**
       * Resizes the machine. 
       */
      void resize(size_t height, size_t width);

      /**
       * Returns the height of the filter/input
       */
      inline size_t getHeight () const { return m_W.extent(0); }

      /**
       * Returns the width of the filter/input
       */
      inline size_t getWidth () const { return m_W.extent(1); }

      /**
       * Returns the current variance Ps estimated at each frequency
       */
      inline const blitz::Array<double, 2>& getPs() const
      { return m_Ps; }

       /**
        * Returns the current variance threshold applied to Ps
        */
      inline const double getVarianceThreshold() const
      { return m_variance_threshold; } 

       /**
        * Returns the current variance Pn estimated at each frequency
        */
      inline const double getPn() const
      { return m_Pn; }
  
      /**
       * Returns the current Wiener filter (in the frequency domain). 
       */
      inline const blitz::Array<double, 2>& getW() const 
      { return m_W; }


      /**
       * Sets the current variance Ps estimated at each frequency.
       * This will also update the Wiener filter, using thresholded values.
       */
      void setPs(const blitz::Array<double,2>& Ps);

      /**
       * Sets the current variance threshold to be used.
       * This will also update the Wiener filter
       */
      void setVarianceThreshold(const double variance_threshold)
      { m_variance_threshold = variance_threshold; computeW(); }

      /**
       * Sets the current noise level Pn to be considered.
       * This will update the Wiener filter
       */
      void setPn(const double Pn)
      { m_Pn = Pn; computeW(); }


    private: //representation

      void computeW(); /// Compute the Wiener filter using Pn, Ps, etc. 

      blitz::Array<double, 2> m_Ps; ///< variance at each frequency estimated empirically
      double m_variance_threshold; ///< Threshold on Ps values when computing the Wiener filter
                                   ///  (to avoid division by zero)
      double m_Pn; ///< variance of the noise
      blitz::Array<double, 2> m_W; ///< Wiener filter in the frequency domain (W=1/(1+Pn/Ps))
      boost::shared_ptr<bob::sp::FFT2D> m_fft;
      boost::shared_ptr<bob::sp::IFFT2D> m_ifft;

      mutable blitz::Array<std::complex<double>, 2> m_buffer1; ///< a buffer for speed
      mutable blitz::Array<std::complex<double>, 2> m_buffer2; ///< a buffer for speed
  
  };

}}

#endif /* BOB_MACHINE_WIENERMACHINE_H */
