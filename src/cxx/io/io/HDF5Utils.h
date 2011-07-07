/**
 * @author <a href="mailto:andre.dos.anjos@gmail.com">Andre Anjos</a> 
 * @date Wed  6 Apr 19:41:27 2011 
 *
 * @brief A bunch of private utilities to make programming against the HDF5
 * library a little bit more confortable.
 *
 * Classes and non-member methods in this file handle the low-level HDF5 C-API
 * and try to make it a little bit safer and higher-level for use by the
 * publicly visible HDF5File class. The functionality here is heavily based on
 * boost::shared_ptr's for handling automatic deletion and releasing of HDF5
 * objects. Two top-level classes do the whole work: File and Dataset. The File
 * class represents a raw HDF5 file. You can iterate with it in a very limited
 * way: create one, rename an object or delete one. The Dataset object
 * encapsulates reading and writing of data from a specific HDF5 dataset.
 * Everything is handled automatically and the user should not have to worry
 * about it too much. 
 *
 * @todo Missing support for std::string, list<std::string>
 * @todo Missing support for attributes
 * @todo Missing support for arbitrary groups (80% done see TODOs)
 * @todo Inprint file creation time, author, comments?
 * @todo Missing support for automatic endianness conversion
 * @todo Missing true support for scalars
 */

#ifndef TORCH_IO_HDF5UTILS_H 
#define TORCH_IO_HDF5UTILS_H

#include <map>

#include <boost/make_shared.hpp>
#include <boost/shared_array.hpp>
#include <boost/shared_ptr.hpp>
#include <boost/filesystem.hpp>
#include <blitz/array.h>
#include <blitz/tinyvec-et.h>
#include <hdf5.h>

#include "core/Exception.h"
#include "core/array_assert.h"

#include "io/HDF5Exception.h"
#include "io/HDF5Types.h"

/**
 * Checks if the version of HDF5 installed is greater or equal to some set of
 * values. (extracted from hdf5-1.8.7)
 */
#ifndef H5_VERSION_GE
#define H5_VERSION_GE(Maj,Min,Rel) \
 (((H5_VERS_MAJOR==Maj) && (H5_VERS_MINOR==Min) && (H5_VERS_RELEASE>=Rel)) || \
  ((H5_VERS_MAJOR==Maj) && (H5_VERS_MINOR>Min)) || \
  (H5_VERS_MAJOR>Maj))
#endif

namespace Torch { namespace io { namespace detail { namespace hdf5 {

  //Forward declaration for File
  class Dataset;

  /**
   * An HDF5 C-style file that knows how to close itself.
   */
  class File {

    public:

      /**
       * Creates a new HDF5 file.
       */
      File(const boost::filesystem::path& path, unsigned int flags);

      /**
       * Destructor virtualization
       */
      virtual ~File();

      /**
       * Unlinks a particular dataset from the file. Please note that this will
       * not erase the data on the current file as that functionality is not
       * provided by HDF5. To actually reclaim the space occupied by the
       * unlinked structure, you must re-save this file to another file. The
       * new file will not contain the data of any dangling datasets (datasets
       * w/o names or links).
       */
      void unlink (const std::string& path);

      /**
       * Renames a given dataset or group. Creates intermediary groups if
       * necessary.
       */
      void rename (const std::string& from,
          const std::string& to);

    private: //not implemented

      File(const File& other);

      File& operator= (const File& other);

    public: //representation

      const boost::filesystem::path m_path; ///< path to the file
      unsigned int m_flags; ///< flags used to open it
      boost::shared_ptr<hid_t> m_id; ///< the HDF5 id attributed to this file.
  };

  /**
   * An HDF5 C-style dataset that knows how to close itself.
   */
  class Dataset {

    public:

      /**
       * Creates a new HDF5 dataset by reading its contents from a certain
       * file.
       */
      Dataset(boost::shared_ptr<File>& f, const std::string& path);

      /**
       * Creates a new HDF5 dataset from scratch and inserts it in the given
       * file. If the Dataset already exists on file and the types are
       * compatible, we attach to that type, otherwise, we raise an exception.
       */
      Dataset(boost::shared_ptr<File>& f, const std::string&, 
          const Torch::io::HDF5Type& type);

      /**
       * Destructor virtualization
       */
      virtual ~Dataset();

      /**
       * Returns the number of objects installed at this dataset
       */
      inline size_t size() const { return m_extent[0]; }

      /**
       * Tells if this dataset holds arrays (or scalars)
       */
      inline bool is_array() const { return m_type.is_array(); }

      /**
       * DATA READING FUNCTIONALITY
       */

      /**
       * Reads data from the file into a scalar.
       */
      template <typename T> void read(size_t index, T& value) {
        if (index >= m_extent[0])
          throw Torch::io::HDF5IndexError(m_parent->m_path.string(), 
              m_path, m_extent[0], index);
        if (m_type != Torch::io::HDF5Type(value))
          throw Torch::io::HDF5IncompatibleIO(m_parent->m_path.string(), 
              m_path, m_type.str(), Torch::io::HDF5Type(value).str());
        select(index);
        read(reinterpret_cast<void*>(&value));
      }

      /**
       * Reads data from the file into a scalar (allocated internally).
       */
      template <typename T> T read(size_t index) {
        T retval;
        read(index, retval);
        return retval;
      }

      /**
       * Reads data from the file into a scalar. This is equivalent to using
       * read(0, value).
       */
      template <typename T> void read(T& value) {
        read(0, value);
      }

      /**
       * Reads data from the file into a scalar. This is equivalent to using
       * read(0).
       */
      template <typename T> T read() {
        T retval;
        read(0, retval);
        return retval;
      }

      /**
       * Reads data from the file into a array.
       *
       * @param index Which of the arrays to read in the current dataset, by
       * order
       * @param value The output array data will be stored inside this
       * variable. This variable has to be a zero-based C-style contiguous
       * storage array. If that is not the case, we will raise an exception.
       */
      template <typename T, int N> 
        void readArray(size_t index, blitz::Array<T,N>& value) {
          if (index >= m_extent[0])
            throw Torch::io::HDF5IndexError(m_parent->m_path.string(), 
                m_path, m_extent[0], index);
          if (m_type != Torch::io::HDF5Type(value))
            throw Torch::io::HDF5IncompatibleIO
              (m_parent->m_path.string(), m_path, m_type.str(),
               Torch::io::HDF5Type(value).str());
          Torch::core::array::assertCZeroBaseContiguous(value);
          select(index);
          read(reinterpret_cast<void*>(value.data()));
        }

      /**
       * Reads data from the file into an array allocated dynamically.
       *
       * @param index Which of the arrays to read in the current dataset, by
       * order
       */
      template <typename T, int N> 
        blitz::Array<T,N> readArray(size_t index) {
          blitz::TinyVector<int,N> shape;
          m_type.shape().set(shape);
          blitz::Array<T,N> retval(shape);
          readArray(index, retval);
          return retval;
        }

      /**
       * Reads data from the file into a array. This is equivalent to using
       * readArray(0, value).
       *
       * @param index Which of the arrays to read in the current dataset, by
       * order
       * @param value The output array data will be stored inside this
       * variable. This variable has to be a zero-based C-style contiguous
       * storage array. If that is not the case, we will raise an exception.
       */
      template <typename T, int N> 
        void readArray(blitz::Array<T,N>& value) {
          readArray(0, value);
        }

      /**
       * Reads data from the file into a array. This is equivalent to using
       * readArray(0).
       */
      template <typename T, int N> 
        blitz::Array<T,N> readArray() {
          return readArray<T,N>(0);
        }

      /**
       * DATA WRITING FUNCTIONALITY
       */

      /**
       * Modifies the value of a scalar inside the file.
       */
      template <typename T> void replace(size_t index, const T& value) {
        if (index >= m_extent[0])
          throw Torch::io::HDF5IndexError(m_parent->m_path.string(), 
              m_path, m_extent[0], index);
        if (m_type != Torch::io::HDF5Type(value))
          throw Torch::io::HDF5IncompatibleIO(m_parent->m_path.string(), 
              m_path, m_type.str(), Torch::io::HDF5Type(value).str());
        select(index);
        write(reinterpret_cast<const void*>(&value));
      }

      /**
       * Modifies the value of a scalar inside the file. This is equivalent to
       * using replace(0, value).
       */
      template <typename T> void replace(const T& value) {
        replace(0, value);
      }

      /**
       * Inserts a scalar in the current (existing ;-) dataset. This will
       * trigger writing data to the file.
       */
      template <typename T> void add(const T& value) {
        if (!m_chunked)
          throw Torch::io::HDF5NotExpandible(m_parent->m_path.string(), 
              m_path);
        if (m_type != Torch::io::HDF5Type(value))
          throw Torch::io::HDF5IncompatibleIO(m_parent->m_path.string(), 
              m_path, m_type.str(), Torch::io::HDF5Type(value).str());
        extend();
        select(m_extent[0]-1);
        write(reinterpret_cast<const void*>(&value));
      }

      /**
       * Replaces data at the file using a new array.
       *
       * @param index Which of the arrays to read in the current dataset, by
       * order
       * @param value The output array data will be stored inside this
       * variable. This variable has to be a zero-based C-style contiguous
       * storage array. If that is not the case, we will raise an exception.
       */
      template <typename T, int N> 
        void replaceArray(size_t index, const blitz::Array<T,N>& value) {
          if (index >= m_extent[0])
            throw Torch::io::HDF5IndexError(m_parent->m_path.string(), 
                m_path, m_extent[0], index);
          if (m_type != Torch::io::HDF5Type(value))
            throw Torch::io::HDF5IncompatibleIO
              (m_parent->m_path.string(), 
                m_path, m_type.str(), Torch::io::HDF5Type(value).str());
          select(index);
          if(Torch::core::array::isCContiguous(value)) {
            blitz::Array<T,N> tmp = value.copy();
            write(reinterpret_cast<const void*>(tmp.data()));
          }
          else
            write(reinterpret_cast<const void*>(value.data()));
        }

      /**
       * Replaces data at the file using a new array. This is equivalent to
       * calling replaceArray(0, value). 
       *
       * @param value The output array data will be stored inside this
       * variable. This variable has to be a zero-based C-style contiguous
       * storage array. If that is not the case, we will raise an exception.
       */
      template <typename T, int N> 
        void replaceArray(const blitz::Array<T,N>& value) {
          replaceArray(0, value);
        }

      /**
       * Appends a array in a certain subdirectory of the file. If that
       * subdirectory does not exist, it is created. If the dataset does not
       * exist, it is created, otherwise, we append to it. In this case, the
       * dimensionality of the scalar has to be compatible with the existing
       * dataset shape (or "dataspace" in HDF5 parlance). If you want to do
       * this, first unlink and than use one of the add methods.
       */
      template <typename T, int N> 
        void addArray(const blitz::Array<T,N>& value) {
        if (!m_chunked)
          throw Torch::io::HDF5NotExpandible(m_parent->m_path.string(), 
              m_path);
        if (m_type != Torch::io::HDF5Type(value))
          throw Torch::io::HDF5IncompatibleIO(m_parent->m_path.string(), 
              m_path, m_type.str(), Torch::io::HDF5Type(value).str());
        extend();
        select(m_extent[0]-1);
        if(Torch::core::array::isCContiguous(value)) {
          blitz::Array<T,N> tmp = value.copy();
          write(reinterpret_cast<const void*>(tmp.data()));
        }
        else
          write(reinterpret_cast<const void*>(value.data()));
      }

    private: //not implemented

      Dataset(const Dataset& other);

      Dataset& operator= (const Dataset& other);

    private: //some tricks

      /**
       * Selects a bit of the file to be affected at the next read or write
       * operation. This method encapsulate calls to H5Sselect_hyperslab().
       * The input value "index" is assumed to be safe when this method is
       * called.
       */
      void select (size_t index);

      /**
       * Reads a previously selected area into the given (user) buffer.
       */
      void read (void* buffer);

      /**
       * Writes the contents of a given buffer into the file. The area that the
       * data will occupy should have been selected beforehand.
       */
      void write (const void* buffer);

      /**
       * Extend the dataset with one extra variable.
       */
      void extend ();

    public: //representation
  
      boost::shared_ptr<File> m_parent; ///< my parent file
      std::string m_path; ///< full path to this object
      boost::shared_ptr<hid_t> m_id; ///< the HDF5 Dataset this type points to
      boost::shared_ptr<hid_t> m_dt; ///< the datatype of this Dataset
      boost::shared_ptr<hid_t> m_filespace; ///< the "file" space for this set
      boost::shared_ptr<hid_t> m_memspace; ///< the "memory" space for this set
      Torch::io::HDF5Shape m_extent; ///< the actual shape of this set
      Torch::io::HDF5Shape m_select_offset; ///< hyperslab offset
      Torch::io::HDF5Shape m_select_count; ///< hyperslab count
      bool m_chunked; ///< true if this dataset is expandible.
      Torch::io::HDF5Type m_type; ///< the type information for this set

  };

  /**
   * Scans the input file, fills up a dictionary indicating
   * location/pointer to Dataset capable of reading that location.
   */
  void index(boost::shared_ptr<File>& file,
      std::map<std::string, boost::shared_ptr<Dataset> >& index);

}}}}

#endif /* TORCH_IO_HDF5UTILS_H */
