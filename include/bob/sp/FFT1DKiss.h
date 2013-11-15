/**
 * @file bob/sp/FFT1DKiss.h
 * @date Thu Nov 14 11:58:53 CET 2013
 * @author Laurent El Shafey <Laurent.El-Shafey@idiap.ch>
 *
 * @brief Implement a blitz-based 1D Fast Fourier Transform using kiss FFT
 * functions
 *
 * Copyright (C) 2011-2013 Idiap Research Institute, Martigny, Switzerland
 */

#ifndef BOB_SP_FFT1DKISS_H
#define BOB_SP_FFT1DKISS_H

#include <complex>
#include <blitz/array.h>
#include <boost/shared_ptr.hpp>
#include <bob/sp/kissfft.hh>


namespace bob { namespace sp {
/**
 * @ingroup SP
 * @{
 */

/**
 * @brief This class implements a 1D Discrete Fourier Transform based on 
 * the kiss FFT library. It is used as a base class for FFT1D and
 * IFFT1D classes.
 */
class FFT1DKissAbstract
{
  public:
    /**
     * @brief Constructor
     */
    FFT1DKissAbstract(const size_t length);

    /**
     * @brief Copy constructor
     */
    FFT1DKissAbstract(const FFT1DKissAbstract& other);

    /**
     * @brief Destructor
     */
    virtual ~FFT1DKissAbstract();

    /**
     * @brief Assignment operator
     */
    FFT1DKissAbstract& operator=(const FFT1DKissAbstract& other);

    /**
     * @brief Equal operator
     */
    bool operator==(const FFT1DKissAbstract& other) const;

    /**
     * @brief Not equal operator
     */
    bool operator!=(const FFT1DKissAbstract& other) const;

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
     * Private attributes
     */
    size_t m_length;
};


/**
 * @brief This class implements a direct 1D Discrete Fourier Transform 
 * based on the kiss FFT library
 */
class FFT1DKiss: public FFT1DKissAbstract
{
  public:
    /**
     * @brief Constructor
     */ 
    FFT1DKiss(const size_t length);

    /**
     * @brief Copy constructor
     */
    FFT1DKiss(const FFT1DKiss& other);

    /**
     * @brief Destructor
     */
    virtual ~FFT1DKiss();

    /**
     * @brief Assignment operator
     */
    FFT1DKiss& operator=(const FFT1DKiss& other);

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

    /**
     * @brief kiss_fft instance
     */
    boost::shared_ptr<kissfft<double> > m_kissfft;
};


/**
 * @brief This class implements a inverse 1D Discrete Fourier Transform 
 * based on the kiss FFT library
 */
class IFFT1DKiss: public FFT1DKissAbstract
{
  public:
    /**
     * @brief Constructor
     */ 
    IFFT1DKiss(const size_t length);

    /**
     * @brief Copy constructor
     */
    IFFT1DKiss(const IFFT1DKiss& other);

    /**
     * @brief Destructor
     */
    virtual ~IFFT1DKiss();

    /**
     * @brief Assignment operator
     */
    IFFT1DKiss& operator=(const IFFT1DKiss& other);

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

    /**
     * @brief kiss_fft instance
     */
    boost::shared_ptr<kissfft<double> > m_kissfft;
};

/**
 * @}
 */
}}

#endif /* BOB_SP_FFT1DKISS_H */
