/**
 * @file src/core/core/BlitzAdapter.h
 * @author <a href="mailto:Laurent.El-Shafey@idiap.ch">Laurent El Shafey</a> 
 * @author <a href="mailto:andre.anjos@idiap.ch">Andre Anjos</a> 
 *
 * @brief This class defines an adapter for putting and reading Blitz++ 
 * multiarrays from and to streams.
 */

#ifndef TORCH5SPRO_CORE_BLITZ_ADAPTER_H
#define TORCH5SPRO_CORE_BLITZ_ADAPTER_H

#include <iostream>

namespace Torch {
/**
 * \ingroup libcore_api
 * @{
 *
 */
  namespace core {

    /**
     *  @brief The adapter class for blitz++ multiarrays. It adds a header to 
     *  the default Blitz++ i/o stream operators << and >>.
     */
    template<typename T> class BlitzAdapter {

      public:
        /**
         * Constructor
         * 
         * @param blitz_array the Blitz++ array which should be adapted.
         * @param type_checking Set to false will ignore type checking. There 
         *    are two situations where this can be useful: 1/ Blitz++ files 
         *    were saved using a different compiler, resulting in different 
         *    typeid strings. 2/ A conversion is wished such as loading a 
         *    Blitz++<double,2> array from a Blitz++<int,2> stored array. 
         *    Please manage type conversion with EXTREME caution: a/ no OUT 
         *    OF BOUNDARY checks are done. b/ FLOATING POINT TO INTEGER 
         *    conversions are NOT allowed and will results in erratic values.
         */
        BlitzAdapter(T& blitz_array, bool type_checking=true):
          m_array(blitz_array), m_type_checking(type_checking) {}

        /**
         * Write a Blitz++ array into an output stream
         *
         * @param os The output stream where the Blitz++ array should be put
         * @param ad The adapter used to add the header to the Blitz++ array
         */
        friend std::ostream& operator<<(std::ostream& os, 
            const BlitzAdapter<T>& ad)
        {
          // typeid() permits to save a string used to check type consistency 
          // when loading data from a file. However this is compiler-dependent
          // and does not allow to easily reconstruct a blitz array of 
          // unknown type just by reading the stored data.

          // Write header
          os << typeid(T).name() << std::endl;
          os << ad.m_array.dimensions() << std::endl;
          // Write Blitz++ array
          os << ad.m_array << std::endl;
          return os;
        }

        /**
         * Read a Blitz++ array from an input stream
         *
         * @param is The input stream from where to read the Blitz++ array
         * @param ad The adapter used to read the additional header of the 
         *    Blitz++ array
         */
        friend std::istream& operator>>(std::istream& is, BlitzAdapter<T>& ad)
        {
          std::string str_type;
          int dimensions;
          // Read header
          is >> str_type;
          is >> dimensions;
          // Check consistency between the stored data and the given blitz++ 
          // array.
          // TODO: Replace std::cerr with the incoming Warning/Errors 
          // functions and create a new Exception class.
          if( (ad.m_type_checking && str_type.compare(typeid(T).name()) ) || 
              dimensions != ad.m_array.dimensions() )
          {
            std::cerr << "BlitzAdapter::operator>>: Incompatible blitz++ \
              multiarray (type or number of dimensions)." <<std::endl;
            std::exception e;
            throw(e);
          }
          else
            // Read blitz++ array
            is >> ad.m_array;

          return is;
        }
  
      private:
        T& m_array;
        bool m_type_checking;
    };

  }
/**
 * @}
 */
}

#endif /* TORCH5SPRO_CORE_BLITZ_ADAPTER_H */

