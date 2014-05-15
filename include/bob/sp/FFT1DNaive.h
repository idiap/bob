/**
 * @date Wed Apr 13 23:08:13 2011 +0200
 * @author Laurent El Shafey <Laurent.El-Shafey@idiap.ch>
 *
 * @brief Implement a naive 1D Discrete Fourier Transform
 *
 * Copyright (C) Idiap Research Institute, Martigny, Switzerland
 */

#ifndef BOB_SP_FFT1D_NAIVE_H
#define BOB_SP_FFT1D_NAIVE_H

#include <complex>
#include <blitz/array.h>

namespace bob {
/**
 * \ingroup libsp_api
 * @{
 *
 */
namespace sp { namespace detail {

/**
 * @brief This class implements a naive 1D Discrete Fourier Transform.
 */
class FFT1DNaiveAbstract
{
  public:
    /**
     * @brief Constructor: Initialize working array
     */
    FFT1DNaiveAbstract(const size_t length);

    /**
     * @brief Copy constructor
     */
    FFT1DNaiveAbstract(const FFT1DNaiveAbstract& other);

    /**
     * @brief Destructor
     */
    virtual ~FFT1DNaiveAbstract();

    /**
     * @brief Assignment operator
     */
    FFT1DNaiveAbstract& operator=(const FFT1DNaiveAbstract& other);

    /**
     * @brief Equal operator
     */
    bool operator==(const FFT1DNaiveAbstract& other) const;

    /**
     * @brief Not equal operator
     */
    bool operator!=(const FFT1DNaiveAbstract& other) const;

    /**
     * @brief process an array by applying the FFT
     */
    virtual void operator()(const blitz::Array<std::complex<double>,1>& src, 
      blitz::Array<std::complex<double>,1>& dst) = 0;

    /**
     * @brief Reset the FFT1DNaive object for the given 1D shape
     */
    void reset(const size_t length);

    /**
     * @brief Get the current height of the FFT1D object
     */
    size_t getLength() const { return m_length; }
    /**
     * @brief Set the current length of the DCT1D object
     */
    void setLength(const size_t length);

  private:
    /**
     * @brief Initialize the working array
     */
    void initWorkingArray();

    /**
     * @brief Call the initialization procedures
     */
    void reset();

  protected:
    /**
     * Private attributes
     */
    size_t m_length;

    /**
     * Working array
     */
    blitz::Array<std::complex<double>,1> m_wsave; 
};


/**
 * @brief This class implements a naive direct 1D Discrete Fourier 
 * Transform
 */
class FFT1DNaive: public FFT1DNaiveAbstract
{
  public:
    /**
     * @brief Constructor: Initialize working arrays
     */ 
    FFT1DNaive(const size_t length);

    /**
     * @brief Copy constructor
     */
    FFT1DNaive(const FFT1DNaive& other);

    /**
     * @brief Destructor
     */
    virtual ~FFT1DNaive();

    /**
     * @brief process an array by applying the direct FFT
     */
    virtual void operator()(const blitz::Array<std::complex<double>,1>& src, 
      blitz::Array<std::complex<double>,1>& dst);
  
  private:
    /**
     * @brief process an array assuming that all the 'check' are done
     */
    void processNoCheck(const blitz::Array<std::complex<double>,1>& src,
      blitz::Array<std::complex<double>,1>& dst);
};


/**
 * @brief This class implements a naive inverse 1D Discrete Fourier
 * Transform 
 */
class IFFT1DNaive: public FFT1DNaiveAbstract
{
  public:
    /**
     * @brief Constructor: Initialize working array
     */ 
    IFFT1DNaive(const size_t length);

    /**
     * @brief Copy constructor
     */
    IFFT1DNaive(const IFFT1DNaive& other);

    /**
     * @brief Destructor
     */
    virtual ~IFFT1DNaive();

    /**
     * @brief process an array by applying the inverse DFT
     */
    virtual void operator()(const blitz::Array<std::complex<double>,1>& src, 
      blitz::Array<std::complex<double>,1>& dst);

  private:
    /**
     * @brief process an array assuming that all the 'check' are done
     */
    void processNoCheck(const blitz::Array<std::complex<double>,1>& src,
      blitz::Array<std::complex<double>,1>& dst);
};

}}
/**
 * @}
 */
}

#endif /* BOB_SP_FFT1D_NAIVE_H */
