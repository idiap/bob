/**
 * @file database/InlinedArrayImpl.h>
 * @author <a href="mailto:andre.anjos@idiap.ch">Andre Anjos</a> 
 *
 * @brief A class that implements the polimorphic behaviour required when
 * reading and writing blitz arrays to disk or memory.
 */

#ifndef TORCH_DATABASE_INLINEDARRAYIMPL_H
#define TORCH_DATABASE_INLINEDARRAYIMPL_H

#include <blitz/array.h>
#include <core/array_common.h>
#include <database/dataset_common.h>

namespace Torch { namespace database { namespace detail {

  class InlinedArrayImpl {

    public:
     
      /**
       * Starts me with new arbitrary data. Please note we refer to the given
       * array. External modifications to the array memory will affect me. If
       * you don't want that to be the case, make a copy first.
       */
      template<typename T, int D> InlinedArrayImpl(blitz::Array<T,D>& data);

      /**
       * Copy construct by getting an extra reference to somebodies' array.
       */
      InlinedArrayImpl(const InlinedArrayImpl& other);

      /**
       * Destroyes me
       */
      virtual ~InlinedArrayImpl();

      /**
       * Copies the content of the other array and gets a reference to the
       * other array's data.
       */
      InlinedArrayImpl& operator= (const InlinedArrayImpl& other);

      /**
       * This method will set my internal data to the value you specify. We
       * will do this by referring to the data you gave.
       */
      template<typename T, int D> void set(blitz::Array<T,D>& data);

      /**
       * This method will set my internal data to the value you specify. We
       * will do this by copying the data you gave.
       */
      template<typename T, int D> void setCopy(const blitz::Array<T,D>& data);

      /**
       * This method returns a reference to my internal blitz array.
       * It is the fastest way to get access to my data because it involves
       * no data copying. The only downside is that you need to know the
       * correct type and number of dimensions or I'll throw an exception.
       */
      template<typename T, int D> const blitz::Array<T,D>& get() const;

      /**
       * This method returns a copy of my internal blitz array.
       */
      template<typename T, int D> blitz::Array<T,D> getCopy() const;

      /**
       * This method returns a reference to my internal data (not a
       * copy) in the type you wish. It is the easiest method to use because
       * I'll never throw, no matter which type you want to receive data at.
       * Only get the number of dimensions right!
       */
      template<typename T, int D> const blitz::Array<T,D> cast() const;

      /**
       * This method returns a copy of my internal data (not a
       * copy) in the type you wish. It is the easiest method to use because
       * I'll never throw, no matter which type you want to receive data at.
       * Only get the number of dimensions right!
       */
      template<typename T, int D> blitz::Array<T,D> castCopy() const;

      /**
       * Some informative methods
       */
      Torch::core::array::ElementType getElementType() const 
      { return m_elementtype; }
      size_t getNDim() const { return m_ndim; }

    private: //representation
      Torch::core::array::ElementType m_elementtype; ///< Elements' type
      size_t m_ndim; ///< The number of dimensions
      void* m_bzarray; ///< Pointer to the real data

  };

  template<typename T, int D> 
    InlinedArrayImpl::InlinedArrayImpl (blitz::Array<T,D>& data) {
    set(data);
  }

  template<typename T, int D> void InlinedArrayImpl::set(blitz::Array<T,D>& data) {
    m_elementtype = Torch::core::array::getElementType<T>();
    m_ndim = D;
    m_bzarray = reinterpret_cast<void*>(new blitz::Array<T,D>(data));
  }

  template<typename T, int D> 
    void InlinedArrayImpl::setCopy(const blitz::Array<T,D>& data) {
      set(data.copy());
  }

  template<typename T, int D> const blitz::Array<T,D>& InlinedArrayImpl::get() const {
    if (m_elementtype != Torch::core::array::getElementType<T>()) {
      throw Torch::database::TypeError();
    }
    if (m_ndim != D) {
      throw Torch::database::DimensionError();
    }
    return *reinterpret_cast<blitz::Array<T,D>*>(m_bzarray);
  }

  template<typename T, int D> blitz::Array<T,D> InlinedArrayImpl::getCopy() const {
    return get<T,D>()->copy();
  }

  template<typename T, int D> const blitz::Array<T,D> InlinedArrayImpl::cast() const {
    if (D != m_ndim) throw DimensionError();
    //TODO: Ask LES how to use blitz::complex_cast<>() in this situation...
    switch (m_elementtype) {
      case Torch::core::array::t_bool: 
        return blitz::cast<T>(*static_cast<blitz::Array<bool,D> >(m_bzarray));
      case Torch::core::array::t_int8: 
        return blitz::cast<T>(*static_cast<blitz::Array<int8_t,D> >(m_bzarray));
      case Torch::core::array::t_int16: 
        return blitz::cast<T>(*static_cast<blitz::Array<int16_t,D> >(m_bzarray));
      case Torch::core::array::t_int32: 
        return blitz::cast<T>(*static_cast<blitz::Array<int32_t,D> >(m_bzarray));
      case Torch::core::array::t_int64: 
        return blitz::cast<T>(*static_cast<blitz::Array<int64_t,D> >(m_bzarray));
      case Torch::core::array::t_uint8: 
        return blitz::cast<T>(*static_cast<blitz::Array<uint8_t,D> >(m_bzarray));
      case Torch::core::array::t_uint16: 
        return blitz::cast<T>(*static_cast<blitz::Array<uint16_t,D> >(m_bzarray));
      case Torch::core::array::t_uint32: 
        return blitz::cast<T>(*static_cast<blitz::Array<uint32_t,D> >(m_bzarray));
      case Torch::core::array::t_uint64: 
        return blitz::cast<T>(*static_cast<blitz::Array<uint64_t,D> >(m_bzarray));
      case Torch::core::array::t_float32: 
        return blitz::cast<T>(*static_cast<blitz::Array<float,D> >(m_bzarray));
      case Torch::core::array::t_float64: 
        return blitz::cast<T>(*static_cast<blitz::Array<double,D> >(m_bzarray));
      case Torch::core::array::t_float128: 
        return blitz::cast<T>(*static_cast<blitz::Array<long double,D> >(m_bzarray));
      case Torch::core::array::t_complex64: 
        return blitz::cast<T>(*static_cast<blitz::Array<std::complex<float>,D> >(m_bzarray));
      case Torch::core::array::t_complex128: 
        return blitz::cast<T>(*static_cast<blitz::Array<std::complex<double>,D> >(m_bzarray));
      case Torch::core::array::t_complex256: 
        return blitz::cast<T>(*static_cast<blitz::Array<std::complex<long double>,D> >(m_bzarray));
    }

    //if we get to this point, there is nothing much we can do...
    throw TypeError();
  }

  template<typename T, int D> blitz::Array<T,D> InlinedArrayImpl::castCopy() const 
  {
    return cast<T,D>()->copy();
  }

}}}

#endif /* TORCH_DATABASE_INLINEDARRAYIMPL_H */
