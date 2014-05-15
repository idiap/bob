/**
 * @date Thu Apr 7 17:02:42 2011 +0200
 * @author Laurent El Shafey <Laurent.El-Shafey@idiap.ch>
 *
 * @brief Implement a naive 2D Discrete Cosine Transform
 *
 * Copyright (C) Idiap Research Institute, Martigny, Switzerland
 */

#ifndef BOB_SP_DCT2D_NAIVE_H
#define BOB_SP_DCT2D_NAIVE_H

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
class DCT2DNaiveAbstract
{
  public:
    /**
     * @brief Constructor: Initialize working arrays
     */
    DCT2DNaiveAbstract(const size_t height, const size_t width);

    /**
     * @brief Copy constructor
     */
    DCT2DNaiveAbstract(const DCT2DNaiveAbstract& other);

    /**
     * @brief Destructor
     */
    virtual ~DCT2DNaiveAbstract();

    /**
     * @brief Assignment operator
     */
    DCT2DNaiveAbstract& operator=(const DCT2DNaiveAbstract& other);

    /**
     * @brief Equal operator
     */
    bool operator==(const DCT2DNaiveAbstract& other) const;

    /**
     * @brief Not equal operator
     */
    bool operator!=(const DCT2DNaiveAbstract& other) const;

    /**
     * @brief process an array by applying the DCT
     */
    virtual void operator()(const blitz::Array<double,2>& src, 
      blitz::Array<double,2>& dst) = 0;

    /**
     * @brief Reset the DCT2DNaive object for the given 2D shape
     */
    void reset(const size_t height, const size_t width);

    /**
     * @brief Get the current height of the DCT2D object
     */
    size_t getHeight() const { return m_height; }
    /**
     * @brief Get the current width of the DCT2D object
     */
    size_t getWidth() const { return m_width; }
    /**
     * @brief Set the current height of the DCT2D object
     */
    void setHeight(const size_t height);
    /**
     * @brief Set the current width of the DCT2D object
     */
    void setWidth(const size_t width);

  private:
    /**
     * @brief Initialize the normalization factors
     */
    void initNormFactors();

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
    blitz::Array<double,1> m_wsave_h; 
    blitz::Array<double,1> m_wsave_w;

    /**
     * Normalization factors
     */
    double m_sqrt_1h;
    double m_sqrt_2h;
    double m_sqrt_1w;
    double m_sqrt_2w;
};


/**
 * @brief This class implements a naive direct 1D Discrete Cosine 
 * Transform
 */
class DCT2DNaive: public DCT2DNaiveAbstract
{
  public:
    /**
     * @brief Constructor: Initialize working arrays
     */ 
    DCT2DNaive(const size_t height, const size_t width);

    /**
     * @brief Copy constructor
     */
    DCT2DNaive(const DCT2DNaive& other);

    /**
     * @brief Destructor
     */
    virtual ~DCT2DNaive();

    /**
     * @brief process an array by applying the direct DCT
     */
    virtual void operator()(const blitz::Array<double,2>& src, 
      blitz::Array<double,2>& dst);
  
  private:
    /**
     * @brief process an array assuming that all the 'check' are done
     */
    void processNoCheck(const blitz::Array<double,2>& src,
      blitz::Array<double,2>& dst);
};


/**
 * @brief This class implements a naive inverse 1D Discrete Cosine 
 * Transform 
 */
class IDCT2DNaive: public DCT2DNaiveAbstract
{
  public:
    /**
     * @brief Constructor: Initialize working array
     */ 
    IDCT2DNaive(const size_t height, const size_t width);

    /**
     * @brief Copy constructor
     */
    IDCT2DNaive(const IDCT2DNaive& other);

    /**
     * @brief Destructor
     */
    virtual ~IDCT2DNaive();

    /**
     * @brief process an array by applying the inverse DCT
     */
    virtual void operator()(const blitz::Array<double,2>& src, 
      blitz::Array<double,2>& dst);

  private:
    /**
     * @brief process an array assuming that all the 'check' are done
     */
    void processNoCheck(const blitz::Array<double,2>& src,
      blitz::Array<double,2>& dst);
};

}}
/**
 * @}
 */
}

#endif /* BOB_SP_DCT2D_NAIVE_H */
