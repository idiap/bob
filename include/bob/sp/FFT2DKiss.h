/**
 * @file bob/sp/FFT2DKiss.h
 * @date Thu Nov 14 17:18:48 CET 2013
 * @author Laurent El Shafey <Laurent.El-Shafey@idiap.ch>
 *
 * @brief Implement a blitz-based 2D Fast Fourier Transform s FFT
 * from a FFT 1D.
 *
 * Copyright (C) 2011-2013 Idiap Research Institute, Martigny, Switzerland
 */

#ifndef BOB_SP_FFT2DKISS_H
#define BOB_SP_FFT2DKISS_H

#include <complex>
#include <blitz/array.h>
#include <bob/sp/FFT1DKiss.h>


namespace bob { namespace sp {
/**
 * @ingroup SP
 * @{
 */

/**
 * @brief This class implements a 1D Discrete Fourier Transform using a
 * FFT1D implementation.
 */
class FFT2DKissAbstract
{
  public:
    /**
     * @brief Constructor
     */
    FFT2DKissAbstract(const size_t height, const size_t width);

    /**
     * @brief Copy constructor
     */
    FFT2DKissAbstract(const FFT2DKissAbstract& other);

    /**
     * @brief Destructor
     */
    virtual ~FFT2DKissAbstract();

    /**
     * @brief Assignment operator
     */
    FFT2DKissAbstract& operator=(const FFT2DKissAbstract& other);

    /**
     * @brief Equal operator
     */
    bool operator==(const FFT2DKissAbstract& other) const;

    /**
     * @brief Not equal operator
     */
    bool operator!=(const FFT2DKissAbstract& other) const;

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
class FFT2DKiss: public FFT2DKissAbstract
{
  public:
    /**
     * @brief Constructor
     */ 
    FFT2DKiss(const size_t height, const size_t width);

    /**
     * @brief Copy constructor
     */
    FFT2DKiss(const FFT2DKiss& other);

    /**
     * @brief Destructor
     */
    virtual ~FFT2DKiss();

    /**
     * @brief Assignment operator
     */
    FFT2DKiss& operator=(const FFT2DKiss& other);

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
    bob::sp::FFT1DKiss m_fft_h;
    bob::sp::FFT1DKiss m_fft_w;
};


/**
 * @brief This class implements a inverse 2D Discrete Fourier Transform using
 * a FFT1D implementation.
 */
class IFFT2DKiss: public FFT2DKissAbstract
{
  public:
    /**
     * @brief Constructor
     */ 
    IFFT2DKiss(const size_t height, const size_t width);

    /**
     * @brief Copy constructor
     */
    IFFT2DKiss(const IFFT2DKiss& other);

    /**
     * @brief Destructor
     */
    virtual ~IFFT2DKiss();

    /**
     * @brief Assignment operator
     */
    IFFT2DKiss& operator=(const IFFT2DKiss& other);

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
    bob::sp::IFFT1DKiss m_ifft_h;
    bob::sp::IFFT1DKiss m_ifft_w;
};

/**
 * @}
 */
}}

#endif /* BOB_SP_FFT2DKISS_H */
