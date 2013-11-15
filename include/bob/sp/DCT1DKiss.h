/**
 * @file bob/sp/DCT1DKiss.h
 * @date Thu Nov 14 18:16:30 CET 2013
 * @author Laurent El Shafey <Laurent.El-Shafey@idiap.ch>
 *
 * @brief Implement a blitz-based 1D Discrete Cosine Transform using a 1D FFT
 * functions
 *
 * Copyright (C) 2011-2013 Idiap Research Institute, Martigny, Switzerland
 */

#ifndef BOB_SP_DCT1DKISS_H
#define BOB_SP_DCT1DKISS_H

#include <complex>
#include <blitz/array.h>
#include <boost/shared_ptr.hpp>
#include <bob/sp/FFT1DKiss.h>


namespace bob { namespace sp {
/**
 * @ingroup SP
 * @{
 */

/**
 * @brief This class implements a 1D Discrete Fourier Transform based on 
 * the kiss DCT library. It is used as a base class for DCT1D and
 * IDCT1D classes.
 */
class DCT1DKissAbstract
{
  public:
    /**
     * @brief Destructor
     */
    virtual ~DCT1DKissAbstract();

    /**
     * @brief Assignment operator
     */
    DCT1DKissAbstract& operator=(const DCT1DKissAbstract& other);

    /**
     * @brief Equal operator
     */
    bool operator==(const DCT1DKissAbstract& other) const;

    /**
     * @brief Not equal operator
     */
    bool operator!=(const DCT1DKissAbstract& other) const;

    /**
     * @brief process an array by applying the DCT
     */
    virtual void operator()(const blitz::Array<double,1>& src, 
      blitz::Array<double,1>& dst) const;

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
     * @brief Constructor
     */
    DCT1DKissAbstract(const size_t length);

    /**
     * @brief Copy constructor
     */
    DCT1DKissAbstract(const DCT1DKissAbstract& other);

    /**
     * @brief process an array assuming that all the 'check' are done
     */
    virtual void processNoCheck(const blitz::Array<double,1>& src,
      blitz::Array<double,1>& dst) const = 0;
    /**
     * @brief Initialize the normalization factors
     */
    virtual void initNormFactors();
    /**
     * @brief initializes the working array of exponentials
     */
    virtual void initWorkingArray() = 0;

    /**
     * Private attributes
     */
    size_t m_length;
    double m_sqrt_1byl;
    double m_sqrt_2byl;
    blitz::Array<std::complex<double>,1> m_working_array;
};


/**
 * @brief This class implements a direct 1D Discrete Fourier Transform 
 * based on the kiss DCT library
 */
class DCT1DKiss: public DCT1DKissAbstract
{
  public:
    /**
     * @brief Constructor
     */ 
    DCT1DKiss(const size_t length);

    /**
     * @brief Copy constructor
     */
    DCT1DKiss(const DCT1DKiss& other);

    /**
     * @brief Destructor
     */
    virtual ~DCT1DKiss();

    /**
     * @brief Assignment operator
     */
    DCT1DKiss& operator=(const DCT1DKiss& other);

    /**
     * @brief Setters
     */
    virtual void setLength(const size_t length);

  private:
    /**
     * @brief initializes the working array of exponentials
     */
    void initWorkingArray();

    /**
     * @brief process an array assuming that all the 'check' are done
     */
    virtual void processNoCheck(const blitz::Array<double,1>& src,
      blitz::Array<double,1>& dst) const;

    /**
     * Private attributes
     */
    bob::sp::FFT1DKiss m_fft;
    mutable blitz::Array<std::complex<double>,1> m_buffer_1;
    mutable blitz::Array<std::complex<double>,1> m_buffer_2;
};


/**
 * @brief This class implements a inverse 1D Discrete Fourier Transform 
 * based on the kiss DCT library
 */
class IDCT1DKiss: public DCT1DKissAbstract
{
  public:
    /**
     * @brief Constructor
     */ 
    IDCT1DKiss(const size_t length);

    /**
     * @brief Copy constructor
     */
    IDCT1DKiss(const IDCT1DKiss& other);

    /**
     * @brief Destructor
     */
    virtual ~IDCT1DKiss();

    /**
     * @brief Assignment operator
     */
    IDCT1DKiss& operator=(const IDCT1DKiss& other);

    /**
     * @brief Setters
     */
    virtual void setLength(const size_t length);

  private:
    /**
     * @brief initializes the working array of exponentials
     */
    void initWorkingArray();

    /**
     * @brief process an array assuming that all the 'check' are done
     */
    virtual void processNoCheck(const blitz::Array<double,1>& src,
      blitz::Array<double,1>& dst) const;

    /**
     * Private attributes
     */
    bob::sp::IFFT1DKiss m_ifft;
    mutable blitz::Array<std::complex<double>,1> m_buffer_1;
    mutable blitz::Array<std::complex<double>,1> m_buffer_2;
};

/**
 * @}
 */
}}

#endif /* BOB_SP_DCT1DKISS_H */
