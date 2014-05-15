/**
 * @date Thu Oct 20 11:25:46 2011 +0200
 * @author Laurent El Shafey <Laurent.El-Shafey@idiap.ch>
 *
 * @brief This file defines deep copy functions for blitz++ arrays
 *
 * Copyright (C) Idiap Research Institute, Martigny, Switzerland
 */

#ifndef BOB_CORE_ARRAY_COPY_H
#define BOB_CORE_ARRAY_COPY_H

#include <blitz/array.h>
#include <map>
#include <vector>

namespace bob {
  namespace core { namespace array {
    /**
     * @ingroup CORE_ARRAY
     * @{
     */

    /**
     * @brief Copies a blitz array like copy() does, but resets the storage
     * ordering.
     */
    template <typename T, int D>
    blitz::Array<T,D> ccopy(const blitz::Array<T,D>& a)
    {
      blitz::Array<T,D> b(a.shape());
      b = a;
      return b;
    }

    /**
     * @brief Copies a std::vector of blitz arrays, making deep copies of the
     * arrays.
     * @warning Previous content of the destination will be erased
     */
    template <typename T, int D>
    void ccopy(const std::vector<blitz::Array<T,D> >& src,
               std::vector<blitz::Array<T,D> >& dst)
    {
      dst.clear(); // makes sure dst is empty
      for(typename std::vector<blitz::Array<T,D> >::const_iterator
            it=src.begin(); it!=src.end(); ++it)
        dst.push_back(ccopy(*it));
    }

    /**
     * @brief Copies a std::map of blitz arrays, making deep copies of the
     * arrays.
     * @warning Previous content of the destination will be erased
     */
    template <typename K, typename T, int D>
    void ccopy(const std::map<K, blitz::Array<T,D> >& src,
               std::map<K, blitz::Array<T,D> >& dst)
    {
      dst.clear(); // makes sure dst is empty
      for(typename std::map<K, blitz::Array<T,D> >::const_iterator
            it=src.begin(); it!=src.end(); ++it)
        dst[it->first].reference(ccopy(it->second));
    }

    /**
     * @}
     */
  }}
}

#endif /* BOB_CORE_ARRAY_COPY_H */
