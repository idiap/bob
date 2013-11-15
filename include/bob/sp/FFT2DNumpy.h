/**
 * @file bob/sp/FFT2DNumpy.h
 * @date Fri Nov 15 10:13:37 CET 2013
 * @author Laurent El Shafey <Laurent.El-Shafey@idiap.ch>
 *
 * @brief Implement a blitz-based 2D Fast Fourier Transform s FFT
 * from a FFT 1D.
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

#ifndef BOB_SP_FFT2DNUMPY_H
#define BOB_SP_FFT2DNUMPY_H

#include <complex>
#include <blitz/array.h>
#include <bob/sp/FFT1DNumpy.h>


namespace bob { namespace sp {
/**
 * @ingroup SP
 * @{
 */

/**
 * @brief This class implements a 1D Discrete Fourier Transform using a
 * FFT1D implementation.
 */
class FFT2DNumpyAbstract
{
  public:
    /**
     * @brief Constructor
     */
    FFT2DNumpyAbstract(const size_t height, const size_t width);

    /**
     * @brief Copy constructor
     */
    FFT2DNumpyAbstract(const FFT2DNumpyAbstract& other);

    /**
     * @brief Destructor
     */
    virtual ~FFT2DNumpyAbstract();

    /**
     * @brief Assignment operator
     */
    FFT2DNumpyAbstract& operator=(const FFT2DNumpyAbstract& other);

    /**
     * @brief Equal operator
     */
    bool operator==(const FFT2DNumpyAbstract& other) const;

    /**
     * @brief Not equal operator
     */
    bool operator!=(const FFT2DNumpyAbstract& other) const;

    /**
     * @brief process an array by applying the FFT
     */
    virtual void operator()(const blitz::Array<std::complex<double>,2>& src, 
      blitz::Array<std::complex<double>,2>& dst) const;

    /**
     * @brief Getters
     */
    size_t getHeight() const { return m_height; }
    size_t getWidth() const { return m_width; }

    /**
     * @brief Setters
     */
    void setHeight(const size_t height);
    void setWidth(const size_t width);

  protected:
    /**
     * @brief process an array assuming that all the 'check' are done
     */
    virtual void processNoCheck(const blitz::Array<std::complex<double>,2>& src,
      blitz::Array<std::complex<double>,2>& dst) const = 0;

    /**
     * Private attributes
     */
    size_t m_height;
    size_t m_width;
    mutable blitz::Array<std::complex<double>,2> m_buffer_hw;
    mutable blitz::Array<std::complex<double>,1> m_buffer_h;
    mutable blitz::Array<std::complex<double>,1> m_buffer_h2;
};


/**
 * @brief This class implements a direct 2D Discrete Fourier Transform using
 * a FFT1D implementation.
 */
class FFT2DNumpy: public FFT2DNumpyAbstract
{
  public:
    /**
     * @brief Constructor
     */ 
    FFT2DNumpy(const size_t height, const size_t width);

    /**
     * @brief Copy constructor
     */
    FFT2DNumpy(const FFT2DNumpy& other);

    /**
     * @brief Destructor
     */
    virtual ~FFT2DNumpy();

    /**
     * @brief Assignment operator
     */
    FFT2DNumpy& operator=(const FFT2DNumpy& other);

    /**
     * @brief Setters
     */
    void setHeight(const size_t height);
    void setWidth(const size_t width);

  private:
    /**
     * @brief process an array assuming that all the 'check' are done
     */
    virtual void processNoCheck(const blitz::Array<std::complex<double>,2>& src,
      blitz::Array<std::complex<double>,2>& dst) const;

    /**
     * @brief FFT1D instances
     */
    bob::sp::FFT1DNumpy m_fft_h;
    bob::sp::FFT1DNumpy m_fft_w;
};


/**
 * @brief This class implements a inverse 2D Discrete Fourier Transform using
 * a FFT1D implementation.
 */
class IFFT2DNumpy: public FFT2DNumpyAbstract
{
  public:
    /**
     * @brief Constructor
     */ 
    IFFT2DNumpy(const size_t height, const size_t width);

    /**
     * @brief Copy constructor
     */
    IFFT2DNumpy(const IFFT2DNumpy& other);

    /**
     * @brief Destructor
     */
    virtual ~IFFT2DNumpy();

    /**
     * @brief Assignment operator
     */
    IFFT2DNumpy& operator=(const IFFT2DNumpy& other);

    /**
     * @brief Setters
     */
    void setHeight(const size_t height);
    void setWidth(const size_t width);

  private:
    /**
     * @brief process an array assuming that all the 'check' are done
     */
    virtual void processNoCheck(const blitz::Array<std::complex<double>,2>& src,
      blitz::Array<std::complex<double>,2>& dst) const;

    /**
     * @brief IFFT1D instances
     */
    bob::sp::IFFT1DNumpy m_ifft_h;
    bob::sp::IFFT1DNumpy m_ifft_w;
};

/**
 * @}
 */
}}

#endif /* BOB_SP_FFT2DNUMPY_H */
