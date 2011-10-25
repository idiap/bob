/**
 * @author Laurent El Shafey <Laurent.El-Shafey@idiap.ch>
 * @author Andre Anjos <andre.anjos@idiap.ch>
 * @date Fri 21 Oct 16:00:07 2011 CEST
 *
 * @brief This class can be used to load and store arrays from/to .tensor files
 */

#ifndef TORCH_IO_TENSORFILE_H
#define TORCH_IO_TENSORFILE_H

#include "core/cast.h"
#include "io/TensorFileHeader.h"
#include "io/buffer.h"
#include "io/carray.h"
#include "io/utils.h"
#include "io/Exception.h"

namespace Torch { namespace io {

  /**
   * Defines the flags that might be used when loading/storing a file
   * containing blitz arrays.
   */
  enum _TensorFileFlag {
    _unset   = 0,
    _append  = 1L << 0,
    _in      = 1L << 3,
    _out     = 1L << 4
  };

  /**
   * This class can be used for loading and storing multiarrays from/to
   * tensor files
   */
  class TensorFile
  {
    public:
      /**
       * Defines the bitmask type for providing information about the type of
       * the stream.
       */
      typedef _TensorFileFlag openmode;
      static const openmode append  = _append;
      static const openmode in      = _in;
      static const openmode out     = _out; 

      /**
       * Constructor
       */
      TensorFile(const std::string& filename, openmode f);

      /**
       * Destructor
       */
      ~TensorFile();

      /**
       * Tests if next operation will succeed.
       */
      inline bool operator!() const { return !m_stream; }

      /**
       * Closes the TensorFile
       */
      void close();

      /** 
       * Puts an Array of a given type into the output stream/file. If the
       * type/shape have not yet been set, it is set according to the type
       * and shape given in the blitz array, otherwise the type/shape should
       * match or an exception is thrown.
       *
       * Please note that blitz::Array<> will be implicitly constructed as
       * required and respecting those norms.
       *
       * @warning: Please convert your files to HDF5, this format is
       * deprecated starting on 16.04.2011 - AA
       */
      void write(const buffer& data);

      /**
       * Reads the file data into a buffer - this variant reads the next
       * variable. The buffer size will be reset if required.
       */
      void read(buffer& data); 

      /**
       * Reads the file data into a buffer - this variant allows the
       * specification of a position to read data from. The buffer size will be
       * reset if required.
       */
      void read (size_t index, buffer& data);

      /**
       * Peeks the file and returns the currently set typeinfo
       */
      void peek(typeinfo& info) const;

      /**
       * Gets the number of samples/arrays written so far
       *
       * @warning An exception is thrown if nothing was written so far
       */
      inline size_t size() const { 
        headerInitialized(); 
        return m_n_arrays_written; 
      }

      /**
       * Gets the number of elements per array
       *
       * @warning An exception is thrown if nothing was written so far
       */
      inline size_t getNElements() const { 
        headerInitialized(); 
        return m_header.getNElements(); 
      }

      /**
       * Gets the size along a particular dimension
       *
       * @warning An exception is thrown if nothing was written so far
       */
      inline size_t getSize(size_t dim_index) const { 
        headerInitialized(); 
        return m_header.m_type.shape[dim_index]; 
      }

      /**
       * Initializes the tensor file with the given type and shape.
       */
      inline void initTensorFile(const typeinfo& info) {
        initHeader(info);
      }

    private: //Some stuff I need privately

      /**
       * Checks if the end of the tensor file is reached
       */
      inline void endOfFile() {
        if(m_current_array >= m_header.m_n_samples ) 
          throw IndexError(m_current_array);
      }

      /** 
       * Checks that the header has been initialized, and raise an
       * exception if not
       */
      inline void headerInitialized() const { 
        if (!m_header_init) throw Uninitialized();
      }

      /**
       * Initializes the header of the (output) stream with the given type
       * and shape
       */
      void initHeader(const typeinfo& info);

    public:

      /********************************************************************
       * Specific blitz::Array<> operations
       ********************************************************************/


      /**
       * A shortcut to write a blitz::Array<T,D>
       *
       * @warning: Please convert your files to HDF5, this format is
       * deprecated starting on 16.04.2011 - AA
       */
      template <typename T, int D> 
        inline void write(blitz::Array<T,D>& bz) {
          write(carray(bz));
        }

      /**
       * Load one blitz++ multiarray from the input stream/file All the
       * multiarrays saved have the same dimensions.
       */
      template <typename T, int D> inline blitz::Array<T,D> read() {
        buffer buf;
        read(buf);
        return cast<T,D>(buf);
      }

      template <typename T, int D> inline blitz::Array<T,D> read(size_t
          index) { 
        buffer buf;
        read(index, buf);
        return cast<T,D>(buf);
      }

    private: //representation

      bool m_header_init;
      size_t m_current_array;
      size_t m_n_arrays_written;
      std::fstream m_stream;
      detail::TensorFileHeader m_header;
      openmode m_openmode;
  };

  inline _TensorFileFlag operator&(_TensorFileFlag a, _TensorFileFlag b) { 
    return _TensorFileFlag(static_cast<int>(a) & static_cast<int>(b)); 
  }

  inline _TensorFileFlag operator|(_TensorFileFlag a, _TensorFileFlag b) { 
    return _TensorFileFlag(static_cast<int>(a) | static_cast<int>(b)); 
  }

  inline _TensorFileFlag operator^(_TensorFileFlag a, _TensorFileFlag b) { 
    return _TensorFileFlag(static_cast<int>(a) ^ static_cast<int>(b)); 
  }

  inline _TensorFileFlag& operator|=(_TensorFileFlag& a, _TensorFileFlag b) { 
    return a = a | b; 
  }

  inline _TensorFileFlag& operator&=(_TensorFileFlag& a, _TensorFileFlag b) { 
    return a = a & b; 
  }

  inline _TensorFileFlag& operator^=(_TensorFileFlag& a, _TensorFileFlag b) { 
    return a = a ^ b; 
  }

  inline _TensorFileFlag operator~(_TensorFileFlag a) { 
    return _TensorFileFlag(~static_cast<int>(a)); 
  }

} }

#endif /* TORCH_IO_BINFILE_H */
