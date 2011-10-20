/**
 * @file src/cxx/core/core/array_copy.h
 * @author <a href="mailto:Laurent.El-Shafey@idiap.ch">Laurent El Shafey</a> 
 *
 * @brief This file defines deep copy functions for blitz++ arrays
 */

#ifndef TORCH_CORE_ARRAY_COPY_H
#define TORCH_CORE_ARRAY_COPY_H

#include <blitz/array.h>
#if !defined (HAVE_BLITZ_TINYVEC2_H)
#include <blitz/tinyvec-et.h>
#endif
#include <map>
#include <vector>

namespace Torch {
/**
 * \ingroup libcore_api
 * @{
 *
 */
  namespace core { namespace array {

    /**
     * @brief Copies a blitz array like copy() does, but resets the storage 
     * ordering.
     */
    template <typename T, int D>
    blitz::Array<T,D> ccopy( const blitz::Array<T,D>& a)
    {
      blitz::Array<T,D> b(a.shape());
      b = a;
      return b;
    }

    /**
     * @brief Copies a std::vector of blitz arrays, making deep copies of the 
     * arrays.
     */
    template <typename T, int D>
    void ccopy(const std::vector<blitz::Array<T,D> >& src,
               std::vector<blitz::Array<T,D> >& dst)
    {
      for(typename std::vector<blitz::Array<T,D> >::const_iterator 
            it=src.begin(); it!=src.end(); ++it)
        dst.push_back(ccopy(*it));
    }

    /**
     * @brief Copies a std::map of blitz arrays, making deep copies of the 
     * arrays.
     */
    template <typename K, typename T, int D>
    void ccopy(const std::map<K, blitz::Array<T,D> >& src,
               std::map<K, blitz::Array<T,D> >& dst)
    {
      for(typename std::map<K, blitz::Array<T,D> >::const_iterator 
            it=src.begin(); it!=src.end(); ++it)
        dst[it->first].reference(ccopy(it->second));
    }

  }}
/**
 * @}
 */
}

#endif /* TORCH5SPRO_CORE_ARRAY_COPY_H */
