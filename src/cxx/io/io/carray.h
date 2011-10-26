/**
 * @author Andre Anjos <andre.anjos@idiap.ch>
 * @date Wed  5 Oct 09:39:16 2011 CEST
 *
 * @brief A class that implements the polimorphic behaviour required when
 * reading and writing blitz arrays to disk or memory.
 */

#ifndef TORCH_IO_CARRAY_H
#define TORCH_IO_CARRAY_H

#include <stdexcept>
#include <boost/make_shared.hpp>

#include "io/buffer.h"
#include "io/utils.h"

#include "core/array_check.h"
#include "core/array_type.h"
#include "core/cast.h"

namespace Torch { namespace io {

  class carray: public buffer {

    public:

      /**
       * Starts by refering to the data from another carray.
       */
      carray(boost::shared_ptr<carray> other);

      /**
       * Starts by copying the data from another carray.
       */
      carray(const carray& other);

      /**
       * Starts by refering to the data from another buffer.
       */
      carray(boost::shared_ptr<buffer> other);

      /**
       * Starts by copying the data from another buffer.
       */
      carray(const buffer& other);

      /**
       * Starts with an uninitialized, pre-allocated array.
       */
      carray(const typeinfo& info);

      /**
       * Destroyes me
       */
      virtual ~carray();

      /**
       * Copies the data from another buffer.
       */
      virtual void set(const buffer& other);

      /**
       * Refers to the data of another buffer.
       */
      virtual void set(boost::shared_ptr<buffer> other);

      /**
       * Re-allocates this buffer taking into consideration new requirements.
       * The internal memory should be considered uninitialized.
       */
      virtual void set (const typeinfo& req);

      /**
       * Refers to the data of another carray.
       */
      void set(boost::shared_ptr<carray> other);

      /**
       * Element type
       */
      virtual const typeinfo& type() const { return m_type; }

      /**
       * Borrows a reference from the underlying memory. This means this object
       * continues to be responsible for deleting the memory and you should
       * make sure that it outlives the usage of the returned pointer.
       */
      virtual void* ptr() { return m_ptr; }
      virtual const void* ptr() const { return m_ptr; }


      /******************************************************************
       * Blitz Array specific manipulations
       ******************************************************************/


      /**
       * Starts me with new arbitrary data. Please note we refer to the given
       * array. External modifications to the array memory will affect me. If
       * you don't want that to be the case, use the const variant.
       */
      template <typename T, int N> 
        carray(boost::shared_ptr<blitz::Array<T,N> > data) {
          set(data);
        }

      /**
       * Starts me with new arbitrary data. Please note we copy the given
       * array. External modifications to the array memory will not affect me.
       * If you don't want that to be the case, start with a non-const
       * reference.
       */
      template <typename T, int N> 
        carray(const blitz::Array<T,N>& data) {
          set(data);
        }

      /**
       * This method will set my internal data to the value you specify. We
       * will do this by referring to the data you gave.
       */
      template <typename T, int N>
        void set(boost::shared_ptr<blitz::Array<T,N> > data) {
          
          if (Torch::core::array::getElementType<T>() == 
              Torch::core::array::t_unknown)
            throw std::invalid_argument("unsupported element type on blitz::Array<>");
          if (N > TORCH_MAX_DIM) 
            throw std::invalid_argument("unsupported number of dimensions on blitz::Array<>");

          if (!Torch::core::array::isCContiguous(*data.get()))
            throw std::invalid_argument("cannot buffer'ize non-c contiguous array");

          m_type.set(data);

          m_data = data;
          m_ptr = reinterpret_cast<void*>(data->data());
          m_is_blitz = true;
        }

      /**
       * This method will set my internal data to the value you specify. We
       * will do this by copying the data you gave.
       */
      template <typename T, int N> void set(const blitz::Array<T,N>& data) {
        set(boost::make_shared<blitz::Array<T,N> >(Torch::core::array::ccopy(data)));
      }

      /**
       * This method returns a reference to my internal data.  It is the
       * fastest way to get access to my data because it involves no data
       * copying. This method has two limitations:
       *
       * 1) You need to know the correct type and number of dimensions or I'll
       * throw an exception.
       *
       * 2) If this buffer was started by refering to another buffer's data
       * which is not a carray, an exception will be raised.
       * Unfortunately, blitz::Array<>'s do not offer a management mechanism
       * for tracking external data allocation. The exception can be avoided
       * and the referencing mechanism forced if you set the flag "temporary"
       * to "true". In this mode, this method will always suceed, but the
       * object returned will have its lifetime associated to this buffer. In
       * other words, you should make sure this buffer outlives the returned
       * blitz::Array<T,N>.
       */
      template <typename T, int N> blitz::Array<T,N> get(bool temporary=false) {
        
        if (m_is_blitz) {

          if (!m_data) throw std::runtime_error("empty blitz buffer");
          
          if (m_type.dtype != Torch::core::array::getElementType<T>()) throw std::runtime_error("cannot efficiently retrieve blitz::Array<> from buffer containing a different dtype");
          
          if (m_type.nd != N) throw std::runtime_error("cannot retrieve blitz::Array<> from buffer containing different dimensionality");
          
          return *boost::static_pointer_cast<blitz::Array<T,N> >(m_data).get();
        }

        else {
        
          if (temporary) { //returns a temporary reference
            return Torch::io::wrap<T,N>(*this);
          }
          
          else {
            throw std::runtime_error("cannot get() external non-temporary non-blitz buffer array -- for a temporary object, set temporary=true; if you need the returned object to outlive this buffer; use copy() or cast()");
          }
        }

      }

      /**
       * This method returns a copy to my internal data (not a
       * reference) in the type you wish. It is the easiest method to use
       * because I'll never throw, no matter which type you want to receive
       * data at. Only get the number of dimensions right!
       */
      template <typename T, int N> blitz::Array<T,N> cast() const {
        return Torch::io::cast<T,N>(*this);
      }

    private: //representation

      typeinfo m_type; ///< type information
      void* m_ptr; ///< pointer to the data
      bool m_is_blitz; ///< true if initiated with a blitz::Array<>
      boost::shared_ptr<void> m_data; ///< Pointer to the data owner

  };

}}

#endif /* TORCH_IO_CARRAY_H */
