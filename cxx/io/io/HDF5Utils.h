/**
 * @file cxx/io/io/HDF5Utils.h
 * @date Wed Jun 22 17:50:08 2011 +0200
 * @author Andre Anjos <andre.anjos@idiap.ch>
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
 * @todo Inprint file creation time, author, comments?
 * @todo Missing support for automatic endianness conversion
 * @todo Missing true support for scalars
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

#ifndef BOB_IO_HDF5UTILS_H 
#define BOB_IO_HDF5UTILS_H

#include <boost/filesystem.hpp>
#include <boost/shared_ptr.hpp>
#include <boost/enable_shared_from_this.hpp>
#include <hdf5.h>
#include "io/HDF5Group.h"

namespace bob { namespace io { namespace detail { namespace hdf5 {

  /**
   * An HDF5 C-style file that knows how to close itself.
   */
  class File: public boost::enable_shared_from_this<File> {

    public:

      /**
       * Creates a new HDF5 file. Optionally set the userblock size (multiple
       * of 2 number of bytes).
       */
      File(const boost::filesystem::path& path, unsigned int flags,
          size_t userblock_size=0);

      /**
       * Copies a file by creating a copy of each of its groups
       */
      File(const File& other);

      /**
       * Destructor virtualization
       */
      virtual ~File();

      /**
       * Assignment
       */
      File& operator= (const File& other);

      /**
       * Accesses the current location id of this file
       */
      const boost::shared_ptr<hid_t> location() const {
        return m_id;
      }
      boost::shared_ptr<hid_t> location() {
        return m_id;
      }

      /**
       * Returns the userblock size
       */
      size_t userblock_size() const;

      /**
       * Copies the userblock into a string -- not yet implemented. If you want
       * to do it, read the code for the command-line utilitlies h5jam and
       * h5unjam.
       */
      void get_userblock(std::string& data) const;

      /**
       * Writes new data to the user block. Data is truncated up to the size
       * set during file creation -- not yet implemented. If you want to do it,
       * read the code for the command-line utilitlies h5jam and h5unjam.
       */
      void set_userblock(const std::string& data);

      /**
       * Gets the current path
       */
      const std::string& filename() const {
        return m_path.string();
      }

      /**
       * Returns the root group
       */
      boost::shared_ptr<RootGroup> root();

      /**
       * Resets this file, sets to read again all groups and datasets
       */
      void reset();

      /**
       * Tells if this file is writeable
       */
      bool writeable() const;

    private: //representation

      const boost::filesystem::path m_path; ///< path to the file
      unsigned int m_flags; ///< flags used to open it
      boost::shared_ptr<hid_t> m_fcpl; ///< file creation property lists
      boost::shared_ptr<hid_t> m_id; ///< the HDF5 id attributed to this file.
      boost::shared_ptr<RootGroup> m_root;
  };

}}}}

#endif /* BOB_IO_HDF5UTILS_H */
