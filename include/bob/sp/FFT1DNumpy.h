/**
 * @file bob/sp/FFT1DNumpy.h
 * @date Thu Nov 14 11:58:53 CET 2013
 * @author Laurent El Shafey <Laurent.El-Shafey@idiap.ch>
 *
 * @brief Implement a blitz-based 1D Fast Fourier Transform using kiss FFT
 * functions
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

#ifndef BOB_SP_FFT1DNUMPY_H
#define BOB_SP_FFT1DNUMPY_H

#include <complex>
#include <blitz/array.h>
#include <boost/shared_ptr.hpp>
#include <bob/sp/fftpack.h>


namespace bob { namespace sp {
/**
 * @ingroup SP
 * @{
 */

/**
 * @brief This class implements a 1D Discrete Fourier Transform based on 
 * the NumPY FFT implementation. It is used as a base class for FFT1D and
 * IFFT1D classes.
 */
class FFT1DNumpyAbstract
{
  public:
    /**
     * @brief Constructor
     */
    FFT1DNumpyAbstract(const size_t length);

    /**
     * @brief Copy constructor
     */
    FFT1DNumpyAbstract(const FFT1DNumpyAbstract& other);

    /**
     * @brief Destructor
     */
    virtual ~FFT1DNumpyAbstract();

    /**
     * @brief Assignment operator
     */
    FFT1DNumpyAbstract& operator=(const FFT1DNumpyAbstract& other);

    /**
     * @brief Equal operator
     */
    bool operator==(const FFT1DNumpyAbstract& other) const;

    /**
     * @brief Not equal operator
     */
    bool operator!=(const FFT1DNumpyAbstract& other) const;

    /**
     * @brief process an array by applying the FFT
     */
    virtual void operator()(const blitz::Array<std::complex<double>,1>& src, 
      blitz::Array<std::complex<double>,1>& dst) const;

    /**
     * @brief Getters
     */
    size_t getLength() const { return m_length; }
    /**
     * @brief Setters
     */
    virtual void setLength(const size_t length);

  protected:
    /**
     * @brief process an array assuming that all the 'check' are done
     */
    virtual void processNoCheck(const blitz::Array<std::complex<double>,1>& src,
      blitz::Array<std::complex<double>,1>& dst) const = 0;
    /**
     * @brief Initialize working array
     */
    virtual void initWorkingArray();

    /**
     * Private attributes
     */
    size_t m_length;
    blitz::Array<double,1> m_wsave;
    mutable blitz::Array<double,1> m_buffer;
};


/**
 * @brief This class implements a direct 1D Discrete Fourier Transform 
 * based on the NumPy FFT implementation.
 */
class FFT1DNumpy: public FFT1DNumpyAbstract
{
  public:
    /**
     * @brief Constructor
     */ 
    FFT1DNumpy(const size_t length);

    /**
     * @brief Copy constructor
     */
    FFT1DNumpy(const FFT1DNumpy& other);

    /**
     * @brief Destructor
     */
    virtual ~FFT1DNumpy();

    /**
     * @brief Assignment operator
     */
    FFT1DNumpy& operator=(const FFT1DNumpy& other);

    /**
     * @brief Setters
     */
    virtual void setLength(const size_t length);

  private:
    /**
     * @brief process an array assuming that all the 'check' are done
     */
    virtual void processNoCheck(const blitz::Array<std::complex<double>,1>& src,
      blitz::Array<std::complex<double>,1>& dst) const;
};


/**
 * @brief This class implements a inverse 1D Discrete Fourier Transform 
 * based on the NumPy FFT implementation.
 */
class IFFT1DNumpy: public FFT1DNumpyAbstract
{
  public:
    /**
     * @brief Constructor
     */ 
    IFFT1DNumpy(const size_t length);

    /**
     * @brief Copy constructor
     */
    IFFT1DNumpy(const IFFT1DNumpy& other);

    /**
     * @brief Destructor
     */
    virtual ~IFFT1DNumpy();

    /**
     * @brief Assignment operator
     */
    IFFT1DNumpy& operator=(const IFFT1DNumpy& other);

    /**
     * @brief Setters
     */
    virtual void setLength(const size_t length);

  private:
    /**
     * @brief process an array assuming that all the 'check' are done
     */
    virtual void processNoCheck(const blitz::Array<std::complex<double>,1>& src,
      blitz::Array<std::complex<double>,1>& dst) const;
};

/**
 * @}
 */
}}

#endif /* BOB_SP_FFT1DNUMPY_H */
