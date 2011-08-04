/**
 * @author <a href="mailto:andre.anjos@idiap.ch">Andre Anjos</a> 
 * @author <a href="mailto:Laurent.El-Shafey@idiap.ch">Laurent El Shafey</a> 
 * @date Wed 30 Mar 21:06:28 2011 
 *
 * @brief Torch support for HDF5 files. HDF5 is a open standard for
 * self-describing data files. You can get more information in this webpage:
 * http://www.hdfgroup.org/HDF5
 */

#ifndef TORCH_IO_HDF5FILE_H 
#define TORCH_IO_HDF5FILE_H

#include <vector>

#include <boost/ref.hpp>
#include <boost/make_shared.hpp>
#include <boost/tuple/tuple.hpp>
#include <boost/filesystem.hpp>

#include "io/HDF5Utils.h"

namespace Torch { namespace io {

  /**
   * This is the main type for interfacing Torch with HDF5. It allows the user
   * to create, delete and modify data objects using a very-level API. The
   * total functionality provided by this API is, of course, much smaller than
   * what is provided if you use the HDF5 C-APIs directly, but is much simpler
   * as well.
   */
  class HDF5File {

    public:

      /**
       * This enumeration defines the different values with which you can open
       * the files with
       */
      typedef enum mode_t {
        in = 0, //H5F_ACC_RDONLY    < can only read file
        inout = 1, //H5F_ACC_RDWR   < open file for reading and writing
        trunc = 2, //H5F_ACC_TRUNC  < if file exists, truncate it and open
        excl = 4 //H5F_ACC_EXCL    < if file exists, raise, otherwise == inout
      } mode_t;

      /**
       * A precise description of a certain supported type in a dataset.
       *
       * [0] => the dataset type and shape of each element
       * [1] => the number of elements in the dataset
       * [2] => is this dataset expandible using this type?
       */
      typedef boost::tuple<HDF5Type, size_t, bool> description_t;

    public: //api

      /**
       * Constructor, starts a new HDF5File object giving it a file name and an
       * action: excl/trunc/in/inout
       */
      HDF5File (const std::string& filename, mode_t mode);

      /**
       * Destructor virtualization
       */
      virtual ~HDF5File();

      /**
       * Changes the current prefix path. When this object is started, the
       * prefix path is empty, which means all following paths to data objects
       * should be given using the full path. If you set this to a different
       * value, it will be used as a prefix to any subsequent operation until
       * you reset it.
       * 
       * @param path If path starts with '/', it is treated as an absolute
       * path. '..' and '.' are supported. This object should be a std::string.
       * If the value is relative, it is added to the current path. If it is
       * absolute, it causes the prefix to be reset.
       *
       * @note All operations taking a relative path, following a cd(), will be
       * considered relative to the value returned by cwd().
       */
      void cd(const std::string& path);

      /**
       * Returns the current working path, fully resolved
       */
      const std::string& cwd() const;

      /**
       * Tells if we have a variable with the given name inside the HDF5 file.
       * If the file path given is a relative file path, it is taken w.r.t. the
       * current working directory, as returned by cwd().
       */
      bool contains(const std::string& path) const;

      /**
       * Describe a certain dataset path. If the file path is a relative one,
       * it is taken w.r.t. the current working directory, as returned by
       * cwd().
       */
      void describe (const std::string& path, 
          std::vector<description_t>& description) const;

      /**
       * Describes the default binding type for a given dataset.
       */
      void describe (const std::string& path,
          description_t& description) const;

      /**
       * Unlinks a particular dataset from the file. Note that this will
       * not erase the data on the current file as that functionality is not
       * provided by HDF5. To actually reclaim the space occupied by the
       * unlinked structure, you must re-save this file to another file. The
       * new file will not contain the data of any dangling datasets (datasets
       * w/o names or links). Relative paths are allowed.
       */
      void unlink (const std::string& path);

      /**
       * Renames an existing dataset
       */
      void rename (const std::string& from, const std::string& to);

      /**
       * Accesses all existing paths in one shot. Input has to be a std
       * container with T = std::string and accepting push_back()
       */
      template <typename T> void paths (T& container) const {
        for (std::map<std::string, boost::shared_ptr<detail::hdf5::Dataset> >::const_iterator it = m_index.begin(); it != m_index.end(); ++it) {
          container.push_back(it->first);
        }
      }

      /**
       * Copies the contents of the other file to this file. This is a blind
       * operation, so we try to copy everything from the given file to the
       * current one. It is the user responsibility to make sure the "path"
       * slots in the other file are not already taken. If that is detected, an
       * exception will be raised.
       *
       * This operation will be conducted w.r.t. the currently set prefix path
       * (verifiable using cwd()).
       */
      void copy (HDF5File& other);

      /**
       * Reads data from the file into a scalar. Raises an exception if the
       * type is incompatible. Relative paths are accepted.
       */
      template <typename T> void read(const std::string& path, size_t pos, 
          T& value) {
        std::string absolute = resolve(path);
        if (!contains(absolute)) 
          throw Torch::io::HDF5InvalidPath(m_file->m_path.string(), absolute);
        m_index[absolute]->read(pos, value);
      }

      /**
       * Reads data from the file into a scalar. Returns by copy. Raises if the
       * type T is incompatible. Relative paths are accepted.
       */
      template <typename T> T read(const std::string& path, size_t pos) {
        std::string absolute = resolve(path);
        if (!contains(absolute)) 
          throw Torch::io::HDF5InvalidPath(m_file->m_path.string(), absolute);
        return m_index[absolute]->read(pos);
      }

      /**
       * Reads data from the file into a scalar. Raises an exception if the
       * type is incompatible. Relative paths are accepted. Calling this method
       * is equivalent to calling read(path, 0, value).
       */
      template <typename T> void read(const std::string& path, T& value) {
        read(path, 0, value);
      }

      /**
       * Reads data from the file into a scalar. Raises an exception if the
       * type is incompatible. Relative paths are accepted. Calling this method
       * is equivalent to calling read(path, 0). Returns by copy.
       */
      template <typename T> void read(const std::string& path) {
        read<T>(path, 0);
      }

      /**
       * Reads data from the file into a array. Raises an exception if the type
       * is incompatible. Relative paths are accepted.
       */
      template <typename T, int N> void readArray(const std::string& path,
          size_t pos, blitz::Array<T,N>& value) {
        std::string absolute = resolve(path);
        if (!contains(absolute)) 
          throw Torch::io::HDF5InvalidPath(m_file->m_path.string(), absolute);
        m_index[absolute]->readArray(pos, value);
      }

      /**
       * Reads data from the file into a array. Raises an exception if the type
       * is incompatible. Relative paths are accepted. Destination array is
       * allocated internally and returned by value.
       */
      template <typename T, int N> blitz::Array<T,N> readArray
        (const std::string& path, size_t pos) {
        std::string absolute = resolve(path);
        if (!contains(absolute)) 
          throw Torch::io::HDF5InvalidPath(m_file->m_path.string(), absolute);
        return m_index[absolute]->readArray<T,N>(pos);
      }

      /**
       * Reads data from the file into a array. Raises an exception if the type
       * is incompatible. Relative paths are accepted. Calling this method is
       * equivalent to calling readArray(path, 0, value).
       */
      template <typename T, int N> void readArray(const std::string& path,
          blitz::Array<T,N>& value) {
        readArray(path, 0, value);
      }

      /**
       * Reads data from the file into a array. Raises an exception if the type
       * is incompatible. Relative paths are accepted. Calling this method is
       * equivalent to calling readArray(path, 0). Destination array is
       * allocated internally.
       */
      template <typename T, int N> blitz::Array<T,N> readArray
        (const std::string& path) { 
          return readArray<T,N>(path, 0);
      }

      /**
       * Modifies the value of a scalar inside the file. Relative paths are
       * accepted.
       */
      template <typename T> void replace(const std::string& path, size_t pos, 
          const T& value) {
        std::string absolute = resolve(path);
        if (!contains(absolute)) 
          throw Torch::io::HDF5InvalidPath(m_file->m_path.string(), absolute);
        m_index[absolute]->replace(pos, value);
      }

      /**
       * Modifies the value of a scalar inside the file. Relative paths are
       * accepted. Calling this method is equivalent to calling replace(path,
       * 0, value).
       */
      template <typename T> void replace(const std::string& path, 
          const T& value) {
        replace(path, 0, value);
      }

      /**
       * Modifies the value of a array inside the file. Relative paths are
       * accepted.
       */
      template <typename T> void replaceArray(const std::string& path,
          size_t pos, const T& value) {
        std::string absolute = resolve(path);
        if (!contains(absolute)) 
          throw Torch::io::HDF5InvalidPath(m_file->m_path.string(), absolute);
        m_index[absolute]->replaceArray(pos, value);
      }

      /**
       * Modifies the value of a array inside the file. Relative paths are
       * accepted. Calling this method is equivalent to calling
       * replaceArray(path, 0, value).
       */
      template <typename T> void replaceArray(const std::string& path,
          const T& value) {
        replaceArray(path, 0, value);
      }

      /**
       * Appends a scalar in a dataset. If the dataset does not yet exist, one
       * is created with the type characteristics. Relative paths are accepted.
       */
      template <typename T> void append(const std::string& path,
          const T& value) {
        std::string absolute = resolve(path);
        if (!contains(absolute)) { //create dataset
          m_index[absolute] =
            boost::make_shared<detail::hdf5::Dataset>(boost::ref(m_file),
              absolute, Torch::io::HDF5Type(value), true, 0);
        }
        m_index[absolute]->add(value);
      }

      /**
       * Appends a array in a dataset. If the dataset does not yet exist, one
       * is created with the type characteristics. Relative paths are accepted.
       *
       * If a new Dataset is to be created, you can also set the compression
       * level. Note this setting has no effect if the Dataset already exists
       * on file, in which case the current setting for that dataset is
       * respected. The maximum value for the gzip compression is 9. The value
       * of zero turns compression off (the default).
       */
      template <typename T> void appendArray(const std::string& path,
          const T& value, size_t compression=0) {
        std::string absolute = resolve(path);
        if (!contains(absolute)) { //create dataset
          m_index[absolute] =
            boost::make_shared<detail::hdf5::Dataset>(boost::ref(m_file),
              absolute, Torch::io::HDF5Type(value), true, compression);
        }
        m_index[absolute]->addArray(value);
      }

      /**
       * Sets the scalar at position 0 to the given value. This method is
       * equivalent to checking if the scalar at position 0 exists and then
       * replacing it. If the path does not exist, we append the new scalar.
       */
      template <typename T> void set(const std::string& path, const T& value) {
        std::string absolute = resolve(path);
        if (!contains(absolute)) append(path, value);
        else replace(path, value);
      }

      /**
       * Sets the array at position 0 to the given value. This method is
       * equivalent to checking if the array at position 0 exists and then
       * replacing it. If the path does not exist, we append the new array.
       *
       * If a new Dataset is to be created, you can also set the compression
       * level. Note this setting has no effect if the Dataset already exists
       * on file, in which case the current setting for that dataset is
       * respected. The maximum value for the gzip compression is 9. The value
       * of zero turns compression off (the default).
       */
      template <typename T> void setArray(const std::string& path,
          const T& value, size_t compression=0) {
        std::string absolute = resolve(path);
        if (!contains(absolute)) appendArray(path, value, compression);
        else replaceArray(path, value);
      }
      
    private: //not implemented

      /**
       * Copy construct an already opened HDF5File
       */
      HDF5File (const HDF5File& other);

      /**
       * Drop the current settings and load new ones from the other file.
       */
      HDF5File& operator= (const HDF5File& other);

    private: //helpers

      /**
       * Resolves the given path in light of the current prefix set
       * (potentially set with cd()).
       */
      std::string resolve(const std::string& path) const;

    private: //representation

      boost::shared_ptr<detail::hdf5::File> m_file; ///< the opened HDF5 file
      std::map<std::string, boost::shared_ptr<detail::hdf5::Dataset> > m_index; ///< index of datasets currently available on that file
      boost::filesystem::path m_cwd; ///< my current working directory

  };

}}

#endif /* TORCH_IO_HDF5FILE_H */
