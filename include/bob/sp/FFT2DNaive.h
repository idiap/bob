/**
 * @date Wed Apr 13 23:08:13 2011 +0200
 * @author Laurent El Shafey <Laurent.El-Shafey@idiap.ch>
 *
 * @brief Implement a naive 2D Discrete Fourier Transform
 *
 * Copyright (C) Idiap Research Institute, Martigny, Switzerland
 */

#ifndef BOB_SP_FFT2D_NAIVE_H
#define BOB_SP_FFT2D_NAIVE_H

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
class FFT2DNaiveAbstract
{
  public:
    /**
     * @brief Constructor: Initialize working arrays
     */
    FFT2DNaiveAbstract(const size_t height, const size_t width);

    /**
     * @brief Copy constructor
     */
    FFT2DNaiveAbstract(const FFT2DNaiveAbstract& other);

    /**
     * @brief Destructor
     */
    virtual ~FFT2DNaiveAbstract();

    /**
     * @brief Assignment operator
     */
    FFT2DNaiveAbstract& operator=(const FFT2DNaiveAbstract& other);

    /**
     * @brief Equal operator
     */
    bool operator==(const FFT2DNaiveAbstract& other) const;

    /**
     * @brief Not equal operator
     */
    bool operator!=(const FFT2DNaiveAbstract& other) const;

    /**
     * @brief process an array by applying the FFT
     */
    virtual void operator()(const blitz::Array<std::complex<double>,2>& src, 
      blitz::Array<std::complex<double>,2>& dst) = 0;

    /**
     * @brief Reset the FFT2DNaive object for the given 2D shape
     */
    void reset(const size_t height, const size_t width);

    /**
     * @brief Get the current height of the FFT2D object
     */
    size_t getHeight() const { return m_height; }
    /**
     * @brief Get the current width of the FFT2D object
     */
    size_t getWidth() const { return m_width; }
    /**
     * @brief Set the current height of the FFT2D object
     */
    void setHeight(const size_t height);
    /**
     * @brief Set the current width of the FFT2D object
     */
    void setWidth(const size_t width);

  private:
    /**
     * @brief Initialize the working arrays
     */
    void initWorkingArrays();

    /**
     * @brief Call the initialization procedures
     */
    void reset();

  protected:
    /**
     * Private attributes
     */
    size_t m_height;
    size_t m_width;

    /**
     * Working array
     */
    blitz::Array<std::complex<double>,1> m_wsave_h; 
    blitz::Array<std::complex<double>,1> m_wsave_w;
};


/**
 * @brief This class implements a naive direct 1D Discrete Fourier 
 * Transform
 */
class FFT2DNaive: public FFT2DNaiveAbstract
{
  public:
    /**
     * @brief Constructor: Initialize working arrays
     */ 
    FFT2DNaive(const size_t height, const size_t width);

    /**
     * @brief Copy constructor
     */
    FFT2DNaive(const FFT2DNaive& other);

    /**
     * @brief Destructor
     */
    virtual ~FFT2DNaive();

    /**
     * @brief process an array by applying the direct FFT
     */
    virtual void operator()(const blitz::Array<std::complex<double>,2>& src, 
      blitz::Array<std::complex<double>,2>& dst);
  
  private:
    /**
     * @brief process an array assuming that all the 'check' are done
     */
    void processNoCheck(const blitz::Array<std::complex<double>,2>& src,
      blitz::Array<std::complex<double>,2>& dst);
};


/**
 * @brief This class implements a naive inverse 1D Discrete Fourier 
 * Transform 
 */
class IFFT2DNaive: public FFT2DNaiveAbstract
{
  public:
    /**
     * @brief Constructor: Initialize working array
     */ 
    IFFT2DNaive(const size_t height, const size_t width);

    /**
     * @brief Copy constructor
     */
    IFFT2DNaive(const IFFT2DNaive& other);

    /**
     * @brief Destructor
     */
    virtual ~IFFT2DNaive();

    /**
     * @brief process an array by applying the inverse FFT
     */
    virtual void operator()(const blitz::Array<std::complex<double>,2>& src, 
      blitz::Array<std::complex<double>,2>& dst);

  private:
    /**
     * @brief process an array assuming that all the 'check' are done
     */
    void processNoCheck(const blitz::Array<std::complex<double>,2>& src,
      blitz::Array<std::complex<double>,2>& dst);
};

}}
/**
 * @}
 */
}

#endif /* BOB_SP_FFT2D_NAIVE_H */
