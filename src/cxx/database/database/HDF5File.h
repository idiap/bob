/**
 * @author <a href="mailto:andre.anjos@idiap.ch">Andre Anjos</a> 
 * @author <a href="mailto:Laurent.El-Shafey@idiap.ch">Laurent El Shafey</a> 
 * @date Wed 30 Mar 21:06:28 2011 
 *
 * @brief Torch support for HDF5 files. HDF5 is a open standard for
 * self-describing data files. You can get more information in this webpage:
 * http://www.hdfgroup.org/HDF5
 */

#ifndef TORCH_DATABASE_HDF5FILE_H 
#define TORCH_DATABASE_HDF5FILE_H

#include <boost/ref.hpp>

#include "database/HDF5Utils.h"

namespace Torch { namespace database {

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
       * Tells if we have a variable with the given name inside the HDF5 file.
       */
      bool contains(const std::string& path) const;

      /**
       * Describe a certain dataset path
       */
      const HDF5Type& describe (const std::string& path) const;

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
       * Renames an existing dataset
       */
      void rename (const std::string& from, const std::string& to);

      /**
       * Returns the number of elements stored in the given dataset
       */
      size_t size (const std::string& path) const;

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
       * Copies the contents of the current file to another file, already
       * opened. This is a blind operation, so we try to copy everything from
       * the current file to the other file. It is the user responsibility to
       * make sure the "path" slots in the other file are not already taken.
       */
      void copy (HDF5File& other);

      /**
       * Reads data from the file into a scalar. Raises an exception if the
       * type is incompatible.
       */
      template <typename T> void read(const std::string& path, size_t pos, 
          T& value) {
        if (!contains(path)) 
          throw Torch::database::HDF5InvalidPath(m_file->m_path.string(), path);
        m_index[path]->read(pos, value);
      }

      /**
       * Reads data from the file into a array. Raises an exception if the type
       * is incompatible.
       */
      template <typename T> void readArray(const std::string& path, size_t pos, 
          T& value) {
        if (!contains(path)) 
          throw Torch::database::HDF5InvalidPath(m_file->m_path.string(), path);
        m_index[path]->readArray(pos, value);
      }

      /**
       * Modifies the value of a scalar inside the file.
       */
      template <typename T> void replace(const std::string& path, size_t pos, 
          const T& value) {
        if (!contains(path)) 
          throw Torch::database::HDF5InvalidPath(m_file->m_path.string(), path);
        m_index[path]->replace(pos, value);
      }

      /**
       * Modifies the value of a array inside the file.
       */
      template <typename T> void replaceArray(const std::string& path,
          size_t pos, const T& value) {
        if (!contains(path)) 
          throw Torch::database::HDF5InvalidPath(m_file->m_path.string(), path);
        m_index[path]->replaceArray(pos, value);
      }

      /**
       * Appends a scalar in a dataset. If the dataset does not yet exist, one
       * is created with the type characteristics.
       */
      template <typename T> void append(const std::string& path,
          const T& value) {
        if (!contains(path)) { //create dataset
          m_index[path] =
            boost::make_shared<detail::hdf5::Dataset>(boost::ref(m_file),
              path, Torch::database::HDF5Type(value));
        }
        m_index[path]->add(value);
      }

      /**
       * Appends a array in a dataset. If the dataset does not yet exist, one
       * is created with the type characteristics.
       */
      template <typename T> void appendArray(const std::string& path,
          const T& value) {
        if (!contains(path)) { //create dataset
          m_index[path] =
            boost::make_shared<detail::hdf5::Dataset>(boost::ref(m_file),
              path, Torch::database::HDF5Type(value));
        }
        m_index[path]->addArray(value);
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

    private: //representation

      boost::shared_ptr<detail::hdf5::File> m_file; ///< the opened HDF5 file
      std::map<std::string, boost::shared_ptr<detail::hdf5::Dataset> > m_index; ///< index of datasets currently available on that file

  };

}}

#endif /* TORCH_DATABASE_HDF5FILE_H */
