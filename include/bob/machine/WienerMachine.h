/**
 * @file bob/machine/WienerMachine.h
 * @date Fri Sep 30 16:56:06 2011 +0200
 * @author Laurent El Shafey <Laurent.El-Shafey@idiap.ch>
 *
 * Copyright (C) 2011-2013 Idiap Research Institute, Martigny, Switzerland
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

#ifndef BOB_MACHINE_WIENERMACHINE_H
#define BOB_MACHINE_WIENERMACHINE_H

#include "Machine.h"
#include <blitz/array.h>
#include <complex>
#include <bob/io/HDF5File.h>
#include <bob/sp/FFT2D.h>

namespace bob { namespace machine {
/**
 * @ingroup MACHINE
 * @{
 */

/**
 * @brief A Wiener machine, which can be used to denoise a signal,
 * by comparing with a statistical model of the noiseless signal.\n
 * 
 * Reference:\n
 * Computer Vision: Algorithms and Applications, Richard Szeliski
 * (Part 3.4.3)
 */
class WienerMachine: Machine<blitz::Array<double,2>, blitz::Array<double,2> >
{
  public: //api
    /**
     * @brief Default constructor. Builds an otherwise invalid 0 x 0 Wiener 
     * machine. This is equivalent to construct a WienerMachine with two 
     * size_t parameters set to 0, as in WienerMachine(0, 0, 0).
     */
    WienerMachine();

    /**
     * @brief Constructor, builds a new Wiener machine. Wiener filter is
     * initialized with the given size, Ps being sets to the variance 
     * threshold.
     */
    WienerMachine(const size_t height, const size_t width, const double Pn,
      const double variance_threshold=1e-8);

    /**
     * @brief Builds a new machine with the given variance estimate Ps and
     * noise level Pn.
     */
    WienerMachine(const blitz::Array<double,2>& Ps, const double Pn, 
      const double variance_threshold=1e-8);

    /**
     * @brief Builds a new machine with the given Wiener filter. 
     */
    WienerMachine(const blitz::Array<double,2>& W);

    /**
     * @brief Copy constructor
     */
    WienerMachine(const WienerMachine& other);

    /**
     * @brief Starts a new WienerMachine from an existing Configuration
     * object.
     */
    WienerMachine(bob::io::HDF5File& config);

    /**
     * @brief Destructor 
     */
    virtual ~WienerMachine();

    /**
     * @brief Assignment operator
     */
    WienerMachine& operator=(const WienerMachine& other);

    /**
     * @brief Equal to
     */
    bool operator==(const WienerMachine& other) const;

    /**
     * @brief Not equal to
     */
    bool operator!=(const WienerMachine& other) const;

    /**
     * @brief Is similar to
     */
    bool is_similar_to(const WienerMachine& b, const double r_epsilon=1e-5,
      const double a_epsilon=1e-8) const;

    /**
     * @brief Loads data from an existing configuration object. Resets the
     * current state.
     */
    void load(bob::io::HDF5File& config);

    /**
     * @brief Saves an existing machine to a Configuration object.
     */
    void save(bob::io::HDF5File& config) const;

    /**
     * @brief Forwards data through the machine.
     *
     * The input and output are NOT checked for compatibility each time. It
     * is your responsibility to do it.
     */
    void forward_(const blitz::Array<double,2>& input,
        blitz::Array<double,2>& output) const;

    /**
     * @brief Forwards data through the machine.
     *
     * The input and output are checked for compatibility each time the
     * forward method is applied.
     */
    void forward(const blitz::Array<double,2>& input,
        blitz::Array<double,2>& output) const;

    /**
     * @brief Resizes the machine. 
     */
    void resize(const size_t height, const size_t width);

    /**
     * @brief Returns the height of the filter/input
     */
    size_t getHeight() const { return m_W.extent(0); }

    /**
     * @brief Returns the width of the filter/input
     */
    size_t getWidth() const { return m_W.extent(1); }

    /**
     * @brief Returns the current variance Ps estimated at each frequency
     */
    const blitz::Array<double, 2>& getPs() const
    { return m_Ps; }

     /**
      * @brief Returns the current variance threshold applied to Ps
      */
    double getVarianceThreshold() const
    { return m_variance_threshold; } 

     /**
      * @brief Returns the current noise level Pn
      */
    double getPn() const
    { return m_Pn; }

    /**
     * @brief Returns the current Wiener filter (in the frequency domain). 
     */
    const blitz::Array<double, 2>& getW() const 
    { return m_W; }

    /**
     * @brief Sets the height of the filter/input
     */
    void setHeight(const size_t height)
    { resize(height, m_W.extent(1)); }

    /**
     * @brief Returns the width of the filter/input
     */
    void setWidth(const size_t width)
    { resize(m_W.extent(0), width); }

    /**
     * @brief Sets the current variance Ps estimated at each frequency.
     * This will also update the Wiener filter, using thresholded values.
     */
    void setPs(const blitz::Array<double,2>& Ps);

    /**
     * @brief Sets the current variance threshold to be used.
     * This will also update the Wiener filter
     */
    void setVarianceThreshold(const double variance_threshold);

    /**
     * @brief Sets the current noise level Pn to be considered.
     * This will update the Wiener filter
     */
    void setPn(const double Pn)
    { m_Pn = Pn; computeW(); }


  private: //representation
    void computeW(); /// Compute the Wiener filter using Pn, Ps, etc. 
    void applyVarianceThreshold(); /// Apply variance flooring threshold

    blitz::Array<double, 2> m_Ps; ///< variance at each frequency estimated empirically
    double m_variance_threshold; ///< Threshold on Ps values when computing the Wiener filter
                                 ///  (to avoid division by zero)
    double m_Pn; ///< variance of the noise
    blitz::Array<double, 2> m_W; ///< Wiener filter in the frequency domain (W=1/(1+Pn/Ps))
    bob::sp::FFT2D m_fft;
    bob::sp::IFFT2D m_ifft;

    mutable blitz::Array<std::complex<double>, 2> m_buffer1; ///< a buffer for speed
    mutable blitz::Array<std::complex<double>, 2> m_buffer2; ///< a buffer for speed
};

/**
 * @}
 */
}}

#endif /* BOB_MACHINE_WIENERMACHINE_H */
