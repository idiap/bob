/**
 * @file src/core/src/blitz_misc.cc
 * @author <a href="mailto:Laurent.El-Shafey@idiap.ch">Laurent El Shafey</a> 
 *
 * @brief This file contains miscellaneous additions for blitz++ arrays
 *
 * http://www.oonumerics.org/blitz/
 */

#include "core/blitz_misc.h"

BZ_NAMESPACE(blitz)


/** 
 * @brief specialization of operator << for int8_t in 1D
 */
template <>
ostream& operator<<(ostream& os, const Array<int8_t,1>& x)
{
  return out1D_template<int8_t,int16_t>(os,x);
}

/** 
 * @brief specialization of operator << for int8_t in 2D
 */
template <>
ostream& operator<<(ostream& os, const Array<int8_t,2>& x)
{
  return out2D_template<int8_t,int16_t>(os,x);
}

/**
 * @brief specialization of operator << for int8_t in 3D
 */
template <>
ostream& operator<<(ostream& os, const Array<int8_t,3>& x)
{
  return out3D_template<int8_t,int16_t>(os,x);
}

/**
 * @brief specialization of operator << for int8_t in 4D
 */
template <>
ostream& operator<<(ostream& os, const Array<int8_t,4>& x)
{
  return out4D_template<int8_t,int16_t>(os,x);
}


/** 
 * @brief specialization of operator << for uint8_t in 1D
 */
template <>
ostream& operator<<(ostream& os, const Array<uint8_t,1>& x)
{
  return out1D_template<uint8_t,uint16_t>(os,x);
}

/** 
 * @brief specialization of operator << for uint8_t in 2D
 */
template <>
ostream& operator<<(ostream& os, const Array<uint8_t,2>& x)
{
  return out2D_template<uint8_t,uint16_t>(os,x);
}

/**
 * @brief specialization of operator << for uint8_t in 3D
 */
template <>
ostream& operator<<(ostream& os, const Array<uint8_t,3>& x)
{
  return out3D_template<uint8_t,uint16_t>(os,x);
}

/**
 * @brief specialization of operator << for uint8_t in 4D
 */
template <>
ostream& operator<<(ostream& os, const Array<uint8_t,4>& x)
{
  return out4D_template<uint8_t,uint16_t>(os,x);
}


BZ_NAMESPACE_END

