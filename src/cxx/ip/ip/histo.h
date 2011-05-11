#ifndef TORCH5SPRO_IP_HISTO_H
#define TORCH5SPRO_IP_HISTO_H

#include <stdint.h>
#include <blitz/array.h>

#include "core/array_assert.h"
#include "core/array_type.h"

namespace tca = Torch::core::array;
namespace Torch { 
  namespace ip {
    /**
     * This exception is thrown when the histogram computation for a particular type is not implemented in torch
     */
    class UnsupportedTypeForHistogram: public Torch::core::Exception {
    public:
      UnsupportedTypeForHistogram(Torch::core::array::ElementType elementType)  throw();
      UnsupportedTypeForHistogram(const UnsupportedTypeForHistogram& other) throw();
      
      virtual ~UnsupportedTypeForHistogram() throw();
      virtual const char* what() const throw();
    private:
      Torch::core::array::ElementType elementType;
      char description[500];
    };
    
    /**
     * This exception is thrown when a function argument is invalid
     */
    class InvalidArgument: public Torch::core::Exception {
    public:
      InvalidArgument()  throw();
      InvalidArgument(const InvalidArgument& other) throw();
      
      virtual ~InvalidArgument() throw();
      virtual const char* what() const throw();
    };
    
    namespace detail {
      
      /**
       * Return the histogram size for a given type T
       * @warning This function works only for uint8_t and uint16_t, 
       *          otherwise it raises UnsupportedTypeForHistogram exception
       */
      template<typename T>
      int getHistoSize() {
        int histo_size = 0;
        tca::ElementType element_type = Torch::core::array::getElementType<T>();
        switch(element_type) {
          case tca::t_uint8:
            histo_size = 256;
            break;
          case tca::t_uint16:
            histo_size = 65536;
            break;
          default:
            throw UnsupportedTypeForHistogram(element_type);
            break;
        }
        
        return histo_size;
      }
      
      
      template<typename T>
      class ReduceHisto {
        
      public:
        
        //You need the following public typedefs and statics as blitz use them
        //internally.
        typedef T T_sourcetype;
        typedef blitz::Array<uint64_t,1> T_resulttype;
        typedef T_resulttype T_numtype;
        
        static const bool canProvideInitialValue = true;
        static const bool needIndex = false;
        static const bool needInit = false;
        
        ReduceHisto() { 
          histo_size = detail::getHistoSize<T>();
          m_result.resize(histo_size);
          reset(); 
        }
        
        ReduceHisto(blitz::Array<uint64_t, 1> initialValue) {
          histo_size = detail::getHistoSize<T>();
          m_result.resize(histo_size);
          reset(initialValue);
        }
        
        //accumulates, doesn't tell the index position
        inline bool operator()(T x) const {
          m_result(x) ++;
          return true;
        }
        
        //accumulates, tells us the index position
        inline bool operator()(T x, int=0) const {
          m_result(x) ++;
          return true;
        }
        
        //gets the result, tells us how many items we have seen
        inline blitz::Array<uint64_t, 1> result(int count) const {
          return m_result;
        }
        
        void reset() const { 
          m_result = 0;
        }
        
        void reset(blitz::Array<uint64_t, 1> initialValue) {
          m_result = initialValue; 
        }
        
        static const char* name() { 
          return "histo"; 
        }
        
      protected: //representation
        int histo_size;
        mutable blitz::Array<uint64_t, 1> m_result;
      };
    }
  }
}



/**
 * This is the bit that declares the reduction for blitz++
 * Warning: Reductions *must* be declared inside the blitz namespace...
 */
namespace blitz {
  BZ_DECL_ARRAY_FULL_REDUCE(histo, Torch::ip::detail::ReduceHisto)
}

namespace Torch { 
  namespace ip {
    
    
    /**
     * Compute an histogram of a 2D array.
     * 
     * @warning This function only accepts arrays of @c uint8_t or @c uint16_t.
     *          Any other type raises a UnsupportedTypeForHistogram exception
     * 
     * @param src source 2D array
     * @param histo result of the function. This array must have 256 elements
     *              for @c uint8_t or 65536 for @c uint16_t
     * @param accumulate if true the result is added to @c histo
     */
    template<typename T>
    void histogram(blitz::Array<T, 2>& src, blitz::Array<uint64_t, 1>& histo, bool accumulate = false) {
      // GetHistoSize returns an exception if T is not uint8_t or uint16_t
      int histo_size = detail::getHistoSize<T>();
      
      tca::assertSameShape<uint64_t, 1>(histo, blitz::shape(histo_size));
      tca::assertZeroBase<uint64_t, 1>(histo);

      if (accumulate) {
        histo += blitz::histo(src);
      }
      else {
        histo = blitz::histo(src);
      }
    }
    
    /**
     * Compute an histogram of a 2D array.
     * 
     * @warning This function only accepts arrays of int or float (int8, int16,
     *          int32, int64, uint8, uint16, uint32, float32, float64 
     *          and float128)
     *          Any other type raises a UnsupportedTypeForHistogram exception
     * @warning You must have @c min <= @c src(i,j) <= @c max, for every i and j
     * @warning If @c min >= @c max or @c nb_bins == 0, a 
     * 
     * @param src source 2D array
     * @param histo result of the function. This array must have @c nb_bins 
     *              elements
     * @param min least possible value in @c src
     * @param max greatest possible value in @c src
     * @param nb_bins number of bins (must not be zero)
     * @param accumulate if true the result is added to @c histo
     */
    template<typename T>
    void histogram(blitz::Array<T, 2>& src, blitz::Array<uint64_t, 1>& histo, T min, T max, uint32_t nb_bins, bool accumulate = false) {
      tca::ElementType element_type = Torch::core::array::getElementType<T>();
      
      // Check that the given type is supported
      switch (element_type) {
        case tca::t_int8:
        case tca::t_int16:
        case tca::t_int32:
        case tca::t_int64:
        case tca::t_uint8:
        case tca::t_uint16:
        case tca::t_uint32:
        case tca::t_uint64:
        case tca::t_float32:
        case tca::t_float64:
        case tca::t_float128:
          // Valid type
          break;
        default:
          // Invalid type
          throw UnsupportedTypeForHistogram(element_type);
          break;
      }
      
      if (max <= min || nb_bins == 0) {
        throw InvalidArgument();
      }
      
      tca::assertSameShape<uint64_t, 1>(histo, blitz::shape(nb_bins));
      tca::assertZeroBase<uint64_t, 1>(histo);
      
      // Handle the special case nb_bins == 1
      if (nb_bins == 1) {
        if (accumulate) {
          histo(0) += histo.size();
        }
        else {
          histo(0) = histo.size();
        }
        
        return;
      }
      
      T width = max - min;
      T bin_size = width / (nb_bins - 1);

      if (!accumulate) {
        histo = 0;
      }
      
      for(int i = src.lbound(0); i <= src.ubound(0); i++) {
        for(int j = src.lbound(1); j <= src.ubound(1); j++) {
          T element = src(i, j);
          // Convert a value into a bin index
          uint32_t index = (uint32_t)((element - min) / bin_size);
          histo(index) ++;
        }
      }
      
    }
  }
}

#endif /* TORCH5SPRO_IP_HISTO_H */