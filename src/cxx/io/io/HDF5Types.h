/**
 * @author <a href="mailto:andre.dos.anjos@gmail.com">Andre Anjos</a> 
 * @date Wed 13 Apr 18:02:16 2011 
 *
 * @brief A few helpers to handle HDF5 datasets in a more abstract way.
 */

#ifndef TORCH_IO_HDF5TYPES_H
#define TORCH_IO_HDF5TYPES_H

#include <vector>
#include <string>
#include <boost/shared_array.hpp>
#include <boost/shared_ptr.hpp>
#include <blitz/array.h>
#include <blitz/tinyvec-et.h>
#include <hdf5.h>

#include "core/array_type.h"

namespace Torch { namespace io {

  /**
   * Supported types
   */
  typedef enum hdf5type {
    s=0, //std::string
    i8, //int8_t
    i16, //int16_t
    i32, //int32_t
    i64, //int64_t
    u8, //uint8_t
    u16, //uint16_t
    u32, //uint32_t
    u64, //uint64_t
    f32, //float
    f64, //double
    f128, //long double
    c64, //std::complex<float>
    c128, //std::complex<double>
    c256, //std::complex<long double>
    unsupported //this must be last
  } hdf5type;

  /**
   * Converts a hdf5type enumeration into its string representation
   */
  const char* stringize (hdf5type t);

  /**
   * A wrapper to handle the HDF5 C-API error printing in a nicer way...
   */
  class HDF5ErrorStack {

    public: //api

      /**
       * Binds the HDF5Error to the current default error stack.
       */
      HDF5ErrorStack ();

      /**
       * Binds to a specific HDF5 error stack
       */
      HDF5ErrorStack (hid_t stack);

      /**
       * Destructor virtualization.
       */
      virtual ~HDF5ErrorStack();

      /**
       * Returns the currently captured error stack
       */
      inline std::vector<std::string>& get() { return m_err; }

      /**
       * Clears the current error stack
       */
      inline void clear() { m_err.clear(); }

      /**
       * Sets muting
       */
      inline void mute () { m_muted = true; }
      inline void unmute () { m_muted = false; }
      inline bool muted () const { return m_muted; }

    private: //not implemented

      HDF5ErrorStack(const HDF5ErrorStack& other);

      HDF5ErrorStack& operator= (const HDF5ErrorStack& other);

    private: //representation
      hid_t m_stack; ///< the stack I'm observing
      bool m_muted; ///< if I'm currently muted
      std::vector<std::string> m_err; ///< the current captured stack
      herr_t (*m_func)(hid_t, void*); ///< temporary cache
      void* m_client_data; ///< temporary cache

  };

  /**
   * Automatic controller for the HDF5 C-API error logger
   */
  class HDF5Error {

    public: //api
      
      /**
       * Retrieves the current instance of the error handler
       */
      static boost::shared_ptr<HDF5Error> instance();

      /**
       * Accesses the strings method of the error stack
       */
      static inline const std::vector<std::string>& get () {
        return instance()->stack().get();
      }

      /**
       * Accesses the clear method of the error stack
       */
      static inline void clear () { instance()->stack().clear(); }

      /**
       * Mutes the current stack
       */
      static inline void mute () { instance()->stack().mute(); }

      /**
       * Unmutes the current stack
       */
      static inline void unmute () { instance()->stack().unmute(); }

      /**
       * Unmutes the error stack
       */
      virtual ~HDF5Error();

      /**
       * Gets the current stack
       */
      inline HDF5ErrorStack& stack() { return m_error; }

    private: //singleton

      static boost::shared_ptr<HDF5Error> s_instance;

      /**
       * Mutes the default error stack for the current thread
       */
      HDF5Error();

      /**
       * Mutes a specific error stack
       */
      HDF5Error(hid_t stack);

    private: //representation

      HDF5ErrorStack m_error; ///< the muted error stack 

  };

  /**
   * This class defines the shape type: a counter and a variable-size hsize_t
   * array that contains the dimensionality of a certain array.
   */
  class HDF5Shape {

    public: //api

      /**
       * Builds a new shape with a certain size and values
       */
      template <typename T> HDF5Shape(const size_t n, const T* values):
        m_n(n), m_shape(new hsize_t[n]) {
          for (size_t i=0; i<n; ++i) m_shape[i] = values[i];
        }

      /**
       * Builds a new shape with data from a blitz::TinyVector
       */
      template <int N> HDF5Shape(const blitz::TinyVector<int,N>& vec):
        m_n(N), m_shape(new hsize_t[N]) {
          for (size_t i=0; i<N; ++i) m_shape[i] = vec[i];
        }

      /**
       * Allocates the shape space, but fills all with zeros
       */
      HDF5Shape (size_t n);

      /**
       * Default constructor (m_n => 0, no shape)
       */
      HDF5Shape ();

      /**
       * Copy construct the shape
       */
      HDF5Shape (const HDF5Shape& other);

      /**
       * Virtual destructor
       */
      virtual ~HDF5Shape();

      /**
       * Resets this new shape
       */
      HDF5Shape& operator= (const HDF5Shape& other);

      /**
       * Returns the current size of shape
       */
      inline size_t n () const { return m_n; }

      /**
       * Returs a pointer to the first element of the shape
       */
      inline const hsize_t* get() const { return m_shape.get(); }
      inline hsize_t* get() { return m_shape.get(); }

      /**
       * Copies the data from the other HDF5Shape. If the other shape is
       * smaller, will copy up to the number of positions in the other shape,
       * if it is bigger, will copy up to my number of positions.
       */
      void copy(const HDF5Shape& other);
      
      /**
       * Sets a TinyVector with the contents of this shape. If the tinyvector
       * shape is smaller, will copy up to the number of positions in the
       * current shape. If that is bigger, will copy up to my number of
       * positions
       */
      template <int N> void set (blitz::TinyVector<int,N>& v) const {
        if (N >= m_n) for (size_t i=0; i<m_n; ++i) v[i] = m_shape[i]; 
        else for (size_t i=0; i<N; ++i) v[i] = m_shape[i];
      }

      /**
       * Resets the current shape so it becomes zero-sized.
       */
      void reset();

      /**
       * Accesses a certain position of this shape (unchecked!)
       */
      inline hsize_t operator[] (size_t pos) const { return m_shape[pos]; }
      inline hsize_t& operator[] (size_t pos) { return m_shape[pos]; }

      /**
       * Left-shift a number of positions, decreases the total size.
       */
      HDF5Shape& operator <<= (size_t pos);

      /**
       * Right-shift a number of positions, fills positions to the left with
       * zeros. Does not increase total size.
       */
      HDF5Shape& operator >>= (size_t pos);

      /**
       * Returns the product of all dimensions
       */
      hsize_t product() const;

      /**
       * Compares two shapes for equality
       */
      bool operator== (const HDF5Shape& other) const;
      bool operator!= (const HDF5Shape& other) const;

      /**
       * Compares a shape with a TinyVector for equality
       */
      template <int N>
      bool operator== (const blitz::TinyVector<int,N>& other) const {
        if (N != m_n) return false;
        for (size_t i=0; i<m_n; ++i) if (m_shape[i] != other[i]) return false;
        return true;
      }

      template <int N>
      bool operator!= (const blitz::TinyVector<int,N>& other) const {
        return !(*this == other);
      }

      /**
       * Tells if this shape is valid
       */
      inline bool operator! () const { return !m_n; }

      /**
       * Returns a tuple-like string representation for this shape
       */
      std::string str() const;

    private: //representation
      size_t m_n; ///< The number of objects in this shape
      boost::shared_array<hsize_t> m_shape; ///< The actual shape values

  };

  /**
   * Support to compare data types, convert types into runtime equivalents and
   * make our life easier when deciding what to input and output.
   */
  class HDF5Type {

    public:

      /**
       * Specific implementations bind the type T to the support_t enum
       */
#     define DECLARE_SUPPORT(T) HDF5Type(const T& value); 
      DECLARE_SUPPORT(bool)
      DECLARE_SUPPORT(int8_t)
      DECLARE_SUPPORT(int16_t)
      DECLARE_SUPPORT(int32_t)
      DECLARE_SUPPORT(int64_t)
      DECLARE_SUPPORT(uint8_t)
      DECLARE_SUPPORT(uint16_t)
      DECLARE_SUPPORT(uint32_t)
      DECLARE_SUPPORT(uint64_t)
      DECLARE_SUPPORT(float)
      DECLARE_SUPPORT(double)
      DECLARE_SUPPORT(long double)
      DECLARE_SUPPORT(std::complex<float>)
      DECLARE_SUPPORT(std::complex<double>)
      DECLARE_SUPPORT(std::complex<long double>)
      DECLARE_SUPPORT(std::string)
#     undef DECLARE_SUPPORT

#     define DECLARE_SUPPORT(T,N) HDF5Type(const blitz::Array<T,N>& value);

#     define DECLARE_BZ_SUPPORT(T) \
      DECLARE_SUPPORT(T,1) \
      DECLARE_SUPPORT(T,2) \
      DECLARE_SUPPORT(T,3) \
      DECLARE_SUPPORT(T,4)

      DECLARE_BZ_SUPPORT(bool)
      DECLARE_BZ_SUPPORT(int8_t)
      DECLARE_BZ_SUPPORT(int16_t)
      DECLARE_BZ_SUPPORT(int32_t)
      DECLARE_BZ_SUPPORT(int64_t)
      DECLARE_BZ_SUPPORT(uint8_t)
      DECLARE_BZ_SUPPORT(uint16_t)
      DECLARE_BZ_SUPPORT(uint32_t)
      DECLARE_BZ_SUPPORT(uint64_t)
      DECLARE_BZ_SUPPORT(float)
      DECLARE_BZ_SUPPORT(double)
      DECLARE_BZ_SUPPORT(long double)
      DECLARE_BZ_SUPPORT(std::complex<float>)
      DECLARE_BZ_SUPPORT(std::complex<double>)
      DECLARE_BZ_SUPPORT(std::complex<long double>)
#     undef DECLARE_BZ_SUPPORT
#     undef DECLARE_SUPPORT

      /**
       * Creates a HDF5Type from a HDF5 Dataset, Datatype and Dataspace
       */
      HDF5Type(const boost::shared_ptr<hid_t>& type, 
          const HDF5Shape& extents);

      /**
       * Copy construction
       */
      HDF5Type(const HDF5Type& other);

      /**
       * Virtual destructor
       */
      virtual ~HDF5Type();

      /**
       * Assignment
       */
      HDF5Type& operator= (const HDF5Type& other);

      /**
       * Checks if to types are the same
       */
      bool operator== (const HDF5Type& other) const;

      /**
       * Checks if to types are *not* the same
       */
      bool operator!= (const HDF5Type& other) const;

      /**
       * Checks if an existing object is compatible with my type
       */
      template <typename T> bool compatible (const T& value) const {
        return *this == HDF5Type(value);
      }

      /**
       * Tells if we have arrays inside
       */
      inline bool is_array() const { 
        return !(m_shape.n() == 1 && m_shape[0] == 1); 
      }

      /**
       * Returns the HDF5Shape of the arrays, if we have arrays inside.
       */
      const HDF5Shape& shape() const { return m_shape; }

      /**
       * Returns the equivalent HDF5 type info object for this type.
       */
      boost::shared_ptr<hid_t> htype() const;

      /**
       * Returns a string representation of this supported type.
       */
      std::string str() const;

      /**
       * Returns a string representation of the element type.
       */
      std::string type_str() const { return stringize(m_type); }

      /**
       * Returns the current enumeration for the type
       */
      inline hdf5type type() const { return m_type; }

      /**
       * Returns a mapping between the current type and the supported element
       * types in Torch::core::array
       */
      Torch::core::array::ElementType element_type() const;

    private: //representation

      hdf5type m_type; ///< the precise supported type
      HDF5Shape m_shape; ///< what is the shape of the type

  };

}}

#endif /* TORCH_IO_HDF5TYPES_H */
