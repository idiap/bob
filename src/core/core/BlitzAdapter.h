/**
 * @file src/core/core/BlitzAdapter.h
 * @author <a href="mailto:Laurent.El-Shafey@idiap.ch">Laurent El Shafey</a> 
 * @author <a href="mailto:andre.anjos@idiap.ch">Andre Anjos</a> 
 *
 * @brief This class defines an adapter for putting Blitz++ multiarrays in streams.
 */

#ifndef TORCH5SPRO_CORE_BLITZ_ADAPTER_H
#define TORCH5SPRO_CORE_BLITZ_ADAPTER_H

#include <iostream>

namespace Torch {
  namespace core {

    /**
     *  @brief the adapter class with streams for blitz++ multiarrays 
     */
    template<typename T> class BlitzAdapter {

      public:
        /**
         * Constructor
         */
        BlitzAdapter(T& blitz_array): m_array(blitz_array) {}

        /**
         * Write a Blitz++ array into an output stream
         */
        friend std::ostream& operator<<(std::ostream& os, const BlitzAdapter<T>& ad)
        {
          // typeid() permits to save a string used to check type consistency when 
          // loading data from a file. However this is compiler-dependent! and does not 
          // allow to easily reconstruct a blitz array of unknown type just  by reading 
          // the stored data.
          os << typeid(T).name() << std::endl;
          os << ad.m_array.dimensions() << std::endl;
          os << ad.m_array << std::endl;
          return os;
        }

        /**
         * Read a Blitz++ array from an input stream
         */
        friend std::istream& operator>>(std::istream& is, BlitzAdapter<T>& ad)
        {
          std::string str_type;
          int dimensions;
          is >> str_type;
          is >> dimensions;
          // Check consistency between the stored data and the given blitz++ array
          if( str_type.compare(typeid(T).name()) || dimensions != ad.m_array.dimensions() )
          {
            std::cerr << "BlitzAdapter::operator>>: Incompatible blitz++ multiarray \
              (type or number of dimensions)." <<std::endl;
            std::exception e;
            throw(e);
          }
          else
            is >> ad.m_array;

          return is;
        }
  
      private:
        T& m_array;
    };

  }
}

#endif /* TORCH5SPRO_CORE_BLITZ_ADAPTER_H */

