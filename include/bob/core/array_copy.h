/**
 * @file cxx/core/core/array_copy.h
 * @date Thu Oct 20 11:25:46 2011 +0200
 * @author Laurent El Shafey <Laurent.El-Shafey@idiap.ch>
 *
 * @brief This file defines deep copy functions for blitz++ arrays
 *
 * Copyright (C) 2011-2012 Idiap Research Institute, Martigny, Switzerland
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

#ifndef BOB_CORE_ARRAY_COPY_H
#define BOB_CORE_ARRAY_COPY_H

#include "bob/config.h"

#include <blitz/array.h>
#if !defined (HAVE_BLITZ_TINYVEC2_H)
#include <blitz/tinyvec-et.h>
#endif
#include <map>
#include <vector>

namespace bob {
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

  }}
/**
 * @}
 */
}

#endif /* BOB_CORE_ARRAY_COPY_H */
