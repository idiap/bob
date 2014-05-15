/**
 * @date Thu Apr 7 17:02:42 2011 +0200
 * @author Laurent El Shafey <Laurent.El-Shafey@idiap.ch>
 *
 * @brief Implement a naive 1D Discrete Cosine Transform
 *
 * Copyright (C) Idiap Research Institute, Martigny, Switzerland
 */

#ifndef BOB_SP_DCT1D_NAIVE_H
#define BOB_SP_DCT1D_NAIVE_H

#include <blitz/array.h>

namespace bob {
/**
 * \ingroup libsp_api
 * @{
 *
 */
namespace sp { namespace detail {

/**
 * @brief This class implements a naive 1D Discrete Cosine Transform.
 */
class DCT1DNaiveAbstract
{
  public:
    /**
     * @brief Constructor: Initialize working array
     */
    DCT1DNaiveAbstract(const size_t length);

    /**
     * @brief Copy constructor
     */
    DCT1DNaiveAbstract(const DCT1DNaiveAbstract& other);

    /**
     * @brief Destructor
     */
    virtual ~DCT1DNaiveAbstract();

    /**
     * @brief Assignment operator
     */
    DCT1DNaiveAbstract& operator=(const DCT1DNaiveAbstract& other);

    /**
     * @brief Equal operator
     */
    bool operator==(const DCT1DNaiveAbstract& other) const;

    /**
     * @brief Not equal operator
     */
    bool operator!=(const DCT1DNaiveAbstract& other) const;

    /**
     * @brief process an array by applying the DCT
     */
    virtual void operator()(const blitz::Array<double,1>& src, 
      blitz::Array<double,1>& dst) = 0;

    /**
     * @brief Reset the DCT1DNaive object for the given 1D shape
     */
    void reset(const size_t length);

    /**
     * @brief Get the current length of the DCT1D object
     */
    size_t getLength() const { return m_length; }
    /**
     * @brief Set the current length of the DCT1D object
     */
    void setLength(const size_t length);

  private:
    /**
     * @brief Initialize the normalization factors
     */
    void initNormFactors();

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
    blitz::Array<double,1> m_wsave; 

    /**
     * Normalization factors
     */
    double m_sqrt_1l;
    double m_sqrt_2l;
};


/**
 * @brief This class implements a naive direct 1D Discrete Cosine 
 * Transform
 */
class DCT1DNaive: public DCT1DNaiveAbstract
{
  public:
    /**
     * @brief Constructor: Initialize working arrays
     */ 
    DCT1DNaive(const size_t length);

    /**
     * @brief Copy constructor
     */
    DCT1DNaive(const DCT1DNaive& other);

    /**
     * @brief Destructor
     */
    virtual ~DCT1DNaive();

    /**
     * @brief process an array by applying the direct DCT
     */
    virtual void operator()(const blitz::Array<double,1>& src, 
      blitz::Array<double,1>& dst);
  
  private:
    /**
     * @brief process an array assuming that all the 'check' are done
     */
    void processNoCheck(const blitz::Array<double,1>& src,
      blitz::Array<double,1>& dst);
};


/**
 * @brief This class implements a naive inverse 1D Discrete Cosine Transform
 */
class IDCT1DNaive: public DCT1DNaiveAbstract
{
  public:
    /**
     * @brief Constructor: Initialize working array
     */ 
    IDCT1DNaive(const size_t length);

    /**
     * @brief Copy constructor
     */
    IDCT1DNaive(const IDCT1DNaive& other);

    /**
     * @brief Destructor
     */
    virtual ~IDCT1DNaive();

    /**
     * @brief process an array by applying the inverse DCT
     */
    virtual void operator()(const blitz::Array<double,1>& src, 
      blitz::Array<double,1>& dst);

  private:
    /**
     * @brief process an array assuming that all the 'check' are done
     */
    void processNoCheck(const blitz::Array<double,1>& src,
      blitz::Array<double,1>& dst);
};

}}
/**
 * @}
 */
}

#endif /* BOB_SP_DCT1D_NAIVE_H */
