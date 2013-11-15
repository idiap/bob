/**
 * @file bob/sp/FFT2D.h
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

#ifndef BOB_SP_FFT2D_H
#define BOB_SP_FFT2D_H

#include <complex>
#include <blitz/array.h>
#include <bob/sp/FFT1D.h>


namespace bob { namespace sp {
/**
 * @ingroup SP
 * @{
 */

/**
 * @brief This class implements a 1D Discrete Fourier Transform using a
 * FFT1D implementation.
 */
class FFT2DAbstract
{
  public:
    /**
     * @brief Destructor
     */
    virtual ~FFT2DAbstract();

    /**
     * @brief Assignment operator
     */
    FFT2DAbstract& operator=(const FFT2DAbstract& other);

    /**
     * @brief Equal operator
     */
    bool operator==(const FFT2DAbstract& other) const;

    /**
     * @brief Not equal operator
     */
    bool operator!=(const FFT2DAbstract& other) const;

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
    virtual void setHeight(const size_t height);
    virtual void setWidth(const size_t width);
    virtual void setShape(const size_t height, const size_t width);

  protected:
    /**
     * @brief Constructor
     */
    FFT2DAbstract();

    /**
     * @brief Constructor
     */
    FFT2DAbstract(const size_t height, const size_t width);

    /**
     * @brief Copy constructor
     */
    FFT2DAbstract(const FFT2DAbstract& other);

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
class FFT2D: public FFT2DAbstract
{
  public:
    /**
     * @brief Constructor
     */ 
    FFT2D();

    /**
     * @brief Constructor
     */ 
    FFT2D(const size_t height, const size_t width);

    /**
     * @brief Copy constructor
     */
    FFT2D(const FFT2D& other);

    /**
     * @brief Destructor
     */
    virtual ~FFT2D();

    /**
     * @brief Assignment operator
     */
    FFT2D& operator=(const FFT2D& other);

    /**
     * @brief Setters
     */
    void setHeight(const size_t height);
    void setWidth(const size_t width);
    void setShape(const size_t height, const size_t width);

  private:
    /**
     * @brief process an array assuming that all the 'check' are done
     */
    virtual void processNoCheck(const blitz::Array<std::complex<double>,2>& src,
      blitz::Array<std::complex<double>,2>& dst) const;

    /**
     * @brief FFT1D instances
     */
    bob::sp::FFT1D m_fft_h;
    bob::sp::FFT1D m_fft_w;
};


/**
 * @brief This class implements a inverse 2D Discrete Fourier Transform using
 * a FFT1D implementation.
 */
class IFFT2D: public FFT2DAbstract
{
  public:
    /**
     * @brief Constructor
     */ 
    IFFT2D();

    /**
     * @brief Constructor
     */ 
    IFFT2D(const size_t height, const size_t width);

    /**
     * @brief Copy constructor
     */
    IFFT2D(const IFFT2D& other);

    /**
     * @brief Destructor
     */
    virtual ~IFFT2D();

    /**
     * @brief Assignment operator
     */
    IFFT2D& operator=(const IFFT2D& other);

    /**
     * @brief Setters
     */
    void setHeight(const size_t height);
    void setWidth(const size_t width);
    void setShape(const size_t height, const size_t width);

  private:
    /**
     * @brief process an array assuming that all the 'check' are done
     */
    virtual void processNoCheck(const blitz::Array<std::complex<double>,2>& src,
      blitz::Array<std::complex<double>,2>& dst) const;

    /**
     * @brief IFFT1D instances
     */
    bob::sp::IFFT1D m_ifft_h;
    bob::sp::IFFT1D m_ifft_w;
};

/**
 * @}
 */
}}

#endif /* BOB_SP_FFT2D_H */
