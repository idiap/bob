/**
 * @file cxx/io/io/HDF5File.h
 * @date Wed Jun 22 17:50:08 2011 +0200
 * @author Andre Anjos <andre.anjos@idiap.ch>
 *
 * @brief bob support for HDF5 files. HDF5 is a open standard for
 * self-describing data files. You can get more information in this webpage:
 * http://www.hdfgroup.org/HDF5
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

#ifndef BOB_IO_HDF5FILE_H 
#define BOB_IO_HDF5FILE_H

#include <boost/format.hpp>
#include "core/array.h"
#include "io/HDF5Utils.h"

namespace bob { namespace io {

  /**
   * This is the main type for interfacing bob with HDF5. It allows the user
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
       * Changes the current prefix path. When this object is started, it 
       * points to the root of the file. If you set this to a different
       * value, it will be used as a prefix to any subsequent operation on
       * relative paths until you reset it.
       * 
       * @param path If path starts with '/', it is treated as an absolute
       * path. '..' and '.' are supported. This object should be a std::string.
       * If the value is relative, it is added to the current path. 
       *
       * @note All operations taking a relative path, following a cd(), will be
       * considered relative to the value returned by cwd().
       */
      void cd(const std::string& path);

      /**
       * Tells if a certain directory exists in a file.
       */
      bool hasGroup(const std::string& path);

      /**
       * Creates a directory within the file. It is an error to recreate a path
       * that already exists. You can check this with hasGroup()
       */
      void createGroup(const std::string& path);

      /**
       * Tells if there is a version number on the current directory
       */
      bool hasVersion() const;

      /**
       * Reads the version number - works if there is one, otherwise, raises an
       * exception.
       */
      uint64_t getVersion() const;

      /**
       * Sets the version number, overwrites if it already exists
       */
      void setVersion(uint64_t version);

      /**
       * Removes the version number, if one exists
       */
      void removeVersion();

      /**
       * Returns the name of the file currently opened
       */
      const std::string& filename() const { return m_file->filename(); }

      /**
       * Returns the current working path, fully resolved. This is
       * re-calculated every time you call this method.
       */
      std::string cwd() const;

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
      const std::vector<HDF5Descriptor>& describe (const std::string& path) const;

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
      template <typename T> void paths (T& container, const bool relative = false) const {
        m_cwd->dataset_paths(container);
        if (relative){
          const std::string d = cwd();
          const int len = d.length()+1;
          for (typename T::iterator it = container.begin(); it != container.end(); ++it){
            // assert that the string contains the current path
            assert(it->find(d) == 0);
            // subtract current path
            *it = it->substr(len);
          }
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
      template <typename T>
        void read(const std::string& path, size_t pos, T& value) {
          (*m_cwd)[path]->read(pos, value);
        }

      /**
       * Reads data from the file into a scalar. Returns by copy. Raises if the
       * type T is incompatible. Relative paths are accepted.
       */
      template <typename T> T read(const std::string& path, size_t pos) {
        return (*m_cwd)[path]->read<T>(pos);
      }

      /**
       * Reads data from the file into a scalar. Raises an exception if the
       * type is incompatible. Relative paths are accepted. Calling this method
       * is equivalent to calling read(path, 0). Returns by copy.
       */
      template <typename T> T read(const std::string& path) {
        return read<T>(path, 0);
      }

      /**
       * Reads data from the file into a array. Raises an exception if the type
       * is incompatible. Relative paths are accepted.
       */
      template <typename T, int N> void readArray(const std::string& path,
          size_t pos, blitz::Array<T,N>& value) {
        (*m_cwd)[path]->readArray(pos, value);
      }

      /**
       * Reads data from the file into a array. Raises an exception if the type
       * is incompatible. Relative paths are accepted. Destination array is
       * allocated internally and returned by value.
       */
      template <typename T, int N> blitz::Array<T,N> readArray
        (const std::string& path, size_t pos) {
        return (*m_cwd)[path]->readArray<T,N>(pos);
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
        if (!m_file->writeable()) {
          boost::format m("cannot replace value at dataset '%s' at path '%s' of file '%s' because it is not writeable");
          m % path % m_cwd->path() % m_file->filename();
          throw std::runtime_error(m.str().c_str());
        }
        (*m_cwd)[path]->replace(pos, value);
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
        if (!m_file->writeable()) {
          boost::format m("cannot replace array at dataset '%s' at path '%s' of file '%s' because it is not writeable");
          m % path % m_cwd->path() % m_file->filename();
          throw std::runtime_error(m.str().c_str());
        }
        (*m_cwd)[path]->replaceArray(pos, value);
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
        if (!m_file->writeable()) {
          boost::format m("cannot append value to dataset '%s' at path '%s' of file '%s' because it is not writeable");
          m % path % m_cwd->path() % m_file->filename();
          throw std::runtime_error(m.str().c_str());
        }
        if (!contains(path)) m_cwd->create_dataset(path, bob::io::HDF5Type(value), true, 0);
        (*m_cwd)[path]->add(value);
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
        if (!m_file->writeable()) {
          boost::format m("cannot append array to dataset '%s' at path '%s' of file '%s' because it is not writeable");
          m % path % m_cwd->path() % m_file->filename();
          throw std::runtime_error(m.str().c_str());
        }
        if (!contains(path)) m_cwd->create_dataset(path, bob::io::HDF5Type(value), true, compression);
        (*m_cwd)[path]->addArray(value);
      }

      /**
       * Sets the scalar at position 0 to the given value. This method is
       * equivalent to checking if the scalar at position 0 exists and then
       * replacing it. If the path does not exist, we append the new scalar.
       */
      template <typename T> void set(const std::string& path, const T& value) {
        if (!m_file->writeable()) {
          boost::format m("cannot set value at dataset '%s' at path '%s' of file '%s' because it is not writeable");
          m % path % m_cwd->path() % m_file->filename();
          throw std::runtime_error(m.str().c_str());
        }
        if (!contains(path)) m_cwd->create_dataset(path, bob::io::HDF5Type(value), false, 0);
        (*m_cwd)[path]->replace(0, value);
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
        if (!m_file->writeable()) {
          boost::format m("cannot set array at dataset '%s' at path '%s' of file '%s' because it is not writeable");
          m % path % m_cwd->path() % m_file->filename();
          throw std::runtime_error(m.str().c_str());
        }
        if (!contains(path)) m_cwd->create_dataset(path, bob::io::HDF5Type(value), false, compression);
        (*m_cwd)[path]->replaceArray(0, value);
      }

    public: //api shortcuts to deal with buffers -- avoid these at all costs!

      /**
       * creates a new dataset. If the dataset already exists, checks if the
       * existing data is compatible with the required type.
       */
      void create (const std::string& path,
          const bob::core::array::typeinfo& dest, bool list,
          size_t compression);

      /**
       * Reads data from the file into a buffer. The given buffer contains
       * sufficient space to hold the type described in "dest". Raises an
       * exception if the type is incompatible with the expected data in the
       * file. Relative paths are accepted.
       */
      void read_buffer (const std::string& path, size_t pos,
          bob::core::array::interface& b);

      /**
       * writes the contents of a given buffer into the file. the area that the
       * data will occupy should have been selected beforehand.
       */
      void write_buffer (const std::string& path, size_t pos,
          const bob::core::array::interface& b);

      /**
       * extend the dataset with one extra variable.
       */
      void extend_buffer (const std::string& path,
          const bob::core::array::interface& b);

    private: //not implemented

      /**
       * Copy construct an already opened HDF5File
       */
      HDF5File (const HDF5File& other);

      /**
       * Drop the current settings and load new ones from the other file.
       */
      HDF5File& operator= (const HDF5File& other);

    private: //representation

      boost::shared_ptr<detail::hdf5::File> m_file; ///< the file itself
      boost::shared_ptr<detail::hdf5::Group> m_cwd; ///< current working dir

  };

}}

#endif /* BOB_IO_HDF5FILE_H */
