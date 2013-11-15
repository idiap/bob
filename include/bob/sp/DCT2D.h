/**
 * @file bob/sp/DCT2D.h
 * @date Thu Nov 14 22:58:14 CET 2013
 * @author Laurent El Shafey <Laurent.El-Shafey@idiap.ch>
 *
 * @brief Implement a blitz-based 2D Discrete Cosine Transform
 * using a 1D DCT implementation.
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

#ifndef BOB_SP_DCT2D_H
#define BOB_SP_DCT2D_H

#include <complex>
#include <blitz/array.h>
#include <bob/sp/DCT1D.h>


namespace bob { namespace sp {
/**
 * @ingroup SP
 * @{
 */

/**
 * @brief This class implements a 2D Discrete Cosine Transform using a
 * 1D DCT implementation.
 */
class DCT2DAbstract
{
  public:
    /**
     * @brief Destructor
     */
    virtual ~DCT2DAbstract();

    /**
     * @brief Assignment operator
     */
    DCT2DAbstract& operator=(const DCT2DAbstract& other);

    /**
     * @brief Equal operator
     */
    bool operator==(const DCT2DAbstract& other) const;

    /**
     * @brief Not equal operator
     */
    bool operator!=(const DCT2DAbstract& other) const;

    /**
     * @brief process an array by applying the DCT
     */
    virtual void operator()(const blitz::Array<double,2>& src, 
      blitz::Array<double,2>& dst) const;

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
    DCT2DAbstract();

    /**
     * @brief Constructor
     */
    DCT2DAbstract(const size_t height, const size_t width);

    /**
     * @brief Copy constructor
     */
    DCT2DAbstract(const DCT2DAbstract& other);

    /**
     * @brief process an array assuming that all the 'check' are done
     */
    virtual void processNoCheck(const blitz::Array<double,2>& src,
      blitz::Array<double,2>& dst) const = 0;

    /**
     * Private attributes
     */
    size_t m_height;
    size_t m_width;
    mutable blitz::Array<double,2> m_buffer_hw;
    mutable blitz::Array<double,1> m_buffer_h;
    mutable blitz::Array<double,1> m_buffer_h2;
};


/**
 * @brief This class implements a direct 2D Discrete Cosine Transform using
 * a 1D DCT implementation.
 */
class DCT2D: public DCT2DAbstract
{
  public:
    /**
     * @brief Constructor
     */ 
    DCT2D();

    /**
     * @brief Constructor
     */ 
    DCT2D(const size_t height, const size_t width);

    /**
     * @brief Copy constructor
     */
    DCT2D(const DCT2D& other);

    /**
     * @brief Destructor
     */
    virtual ~DCT2D();

    /**
     * @brief Assignment operator
     */
    DCT2D& operator=(const DCT2D& other);

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
    virtual void processNoCheck(const blitz::Array<double,2>& src,
      blitz::Array<double,2>& dst) const;

    /**
     * @brief DCT1D instances
     */
    bob::sp::DCT1D m_dct_h;
    bob::sp::DCT1D m_dct_w;
};


/**
 * @brief This class implements an inverse 2D Discrete Fourier Transform using
 * a inverse 1D DCT implementation.
 */
class IDCT2D: public DCT2DAbstract
{
  public:
    /**
     * @brief Constructor
     */ 
    IDCT2D();

    /**
     * @brief Constructor
     */ 
    IDCT2D(const size_t height, const size_t width);

    /**
     * @brief Copy constructor
     */
    IDCT2D(const IDCT2D& other);

    /**
     * @brief Destructor
     */
    virtual ~IDCT2D();

    /**
     * @brief Assignment operator
     */
    IDCT2D& operator=(const IDCT2D& other);

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
    virtual void processNoCheck(const blitz::Array<double,2>& src,
      blitz::Array<double,2>& dst) const;

    /**
     * @brief IDCT1D instances
     */
    bob::sp::IDCT1D m_idct_h;
    bob::sp::IDCT1D m_idct_w;
};

/**
 * @}
 */
}}

#endif /* BOB_SP_DCT2D_H */
