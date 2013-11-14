/**
 * @file bob/sp/DCT2DKiss.h
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

#ifndef BOB_SP_DCT2DKISS_H
#define BOB_SP_DCT2DKISS_H

#include <complex>
#include <blitz/array.h>
#include <bob/sp/DCT1DKiss.h>


namespace bob { namespace sp {
/**
 * @ingroup SP
 * @{
 */

/**
 * @brief This class implements a 2D Discrete Cosine Transform using a
 * 1D DCT implementation.
 */
class DCT2DKissAbstract
{
  public:
    /**
     * @brief Destructor
     */
    virtual ~DCT2DKissAbstract();

    /**
     * @brief Assignment operator
     */
    DCT2DKissAbstract& operator=(const DCT2DKissAbstract& other);

    /**
     * @brief Equal operator
     */
    bool operator==(const DCT2DKissAbstract& other) const;

    /**
     * @brief Not equal operator
     */
    bool operator!=(const DCT2DKissAbstract& other) const;

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
    void setHeight(const size_t height);
    void setWidth(const size_t width);

  protected:
    /**
     * @brief Constructor
     */
    DCT2DKissAbstract(const size_t height, const size_t width);

    /**
     * @brief Copy constructor
     */
    DCT2DKissAbstract(const DCT2DKissAbstract& other);

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
class DCT2DKiss: public DCT2DKissAbstract
{
  public:
    /**
     * @brief Constructor
     */ 
    DCT2DKiss(const size_t height, const size_t width);

    /**
     * @brief Copy constructor
     */
    DCT2DKiss(const DCT2DKiss& other);

    /**
     * @brief Destructor
     */
    virtual ~DCT2DKiss();

    /**
     * @brief Assignment operator
     */
    DCT2DKiss& operator=(const DCT2DKiss& other);

    /**
     * @brief Setters
     */
    void setHeight(const size_t height);
    void setWidth(const size_t width);

  private:
    /**
     * @brief process an array assuming that all the 'check' are done
     */
    virtual void processNoCheck(const blitz::Array<double,2>& src,
      blitz::Array<double,2>& dst) const;

    /**
     * @brief DCT1D instances
     */
    bob::sp::DCT1DKiss m_dct_h;
    bob::sp::DCT1DKiss m_dct_w;
};


/**
 * @brief This class implements an inverse 2D Discrete Fourier Transform using
 * a inverse 1D DCT implementation.
 */
class IDCT2DKiss: public DCT2DKissAbstract
{
  public:
    /**
     * @brief Constructor
     */ 
    IDCT2DKiss(const size_t height, const size_t width);

    /**
     * @brief Copy constructor
     */
    IDCT2DKiss(const IDCT2DKiss& other);

    /**
     * @brief Destructor
     */
    virtual ~IDCT2DKiss();

    /**
     * @brief Assignment operator
     */
    IDCT2DKiss& operator=(const IDCT2DKiss& other);

    /**
     * @brief Setters
     */
    void setHeight(const size_t height);
    void setWidth(const size_t width);

  private:
    /**
     * @brief process an array assuming that all the 'check' are done
     */
    virtual void processNoCheck(const blitz::Array<double,2>& src,
      blitz::Array<double,2>& dst) const;

    /**
     * @brief IDCT1D instances
     */
    bob::sp::IDCT1DKiss m_idct_h;
    bob::sp::IDCT1DKiss m_idct_w;
};

/**
 * @}
 */
}}

#endif /* BOB_SP_DCT2DKISS_H */
