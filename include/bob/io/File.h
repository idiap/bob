/**
 * @file bob/io/File.h
 * @date Tue Oct 25 23:25:46 2011 +0200
 * @author Andre Anjos <andre.anjos@idiap.ch>
 *
 * @brief Describes a generic API for reading and writing data to external
 * files.
 *
 * Copyright (C) 2011-2013 Idiap Research Institute, Martigny, Switzerland
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

#ifndef BOB_IO_FILE_H 
#define BOB_IO_FILE_H

#include <vector>
#include <string>

#include <boost/shared_ptr.hpp>

#include <bob/core/array.h>
#include <bob/core/blitz_array.h>

/**
 * @addtogroup IO io 
 * @brief I/O module API
 */
namespace bob { 
/**
 * @ingroup IO
 */
namespace io {
  /**
   * @ingroup IO
   * @{
   */
      
  /**
   * @brief Files deal with reading and writing multiple (homogeneous) array
   * data to and from files.
   */
  class File {

    public: //abstract API

      virtual ~File();

      /**
       * The filename this array codec current points to
       */
      virtual const std::string& filename() const =0;

      /**
       * The typeinfo of data within this file, if it is supposed to be read as
       * as a sequence of arrays
       */
      virtual const bob::core::array::typeinfo& type() const =0;

      /**
       * The typeinfo of data within this file, if it is supposed to be read as
       * a single array.
       */
      virtual const bob::core::array::typeinfo& type_all() const =0;

      /**
       * The number of arrays available in this file, if it is supposed to be
       * read as a sequence of arrays.
       */
      virtual size_t size() const =0;

      /**
       * Returns the name of the codec, for compatibility reasons.
       */
      virtual const std::string& name() const =0;

      /**
       * Loads the data of the array into memory. If an index is specified
       * loads the specific array data from the file, otherwise, loads the data
       * at position 0.
       *
       * This method will check to see if the given array has enough space. If
       * that is not the case, it will allocate enough space internally by
       * reseting the input array and putting the data read from the file
       * inside.
       */
      virtual void read(bob::core::array::interface& buffer, size_t index) =0;

      /**
       * Loads all the data available at the file into a single in-memory
       * array.
       *
       * This method will check to see if the given array has enough space. If
       * that is not the case, it will allocate enough space internally by
       * reseting the input array and putting the data read from the file
       * inside.
       */
      virtual void read_all(bob::core::array::interface& buffer) =0;

      /**
       * Appends the given buffer into a file. If the file does not exist,
       * create a new file, else, makes sure that the inserted array respects
       * the previously set file structure.
       *
       * Returns the current position of the newly written array.
       */
      virtual size_t append (const bob::core::array::interface& buffer) =0;

      /**
       * Writes the data from the given buffer into the file and act like it is
       * the only piece of data that will ever be written to such a file. Not
       * more data appending may happen after a call to this method.
       */
      virtual void write (const bob::core::array::interface& buffer) =0;

    public: //blitz::Array specific API

      /**
       * This method returns a copy of the array in the file with the element
       * type you wish (just have to get the number of dimensions right!).
       */
      template <typename T, int N> blitz::Array<T,N> cast(size_t index) {
        bob::core::array::blitz_array tmp(type());
        read(tmp, index);
        return tmp.cast<T,N>();
      }

      /**
       * This method returns a copy of the array in the file with the element
       * type you wish (just have to get the number of dimensions right!).
       *
       * This variant loads all data available into the file in a single array.
       */
      template <typename T, int N> blitz::Array<T,N> cast_all() {
        bob::core::array::blitz_array tmp(type_all());
        read_all(tmp);
        return tmp.cast<T,N>();
      }

      template <typename T, int N> void read(blitz::Array<T,N>& io, 
          size_t index) {
        bob::core::array::blitz_array use_this(io);
        use_this.set(type());
        read(use_this, index);
        io.reference(use_this.get<T,N>());
      }

      template <typename T, int N> blitz::Array<T,N> read(size_t index) {
        bob::core::array::blitz_array tmp(type());
        read(tmp, index);
        return tmp.get<T,N>();
      }

      template <typename T, int N> void read_all(blitz::Array<T,N>& io) {
        bob::core::array::blitz_array use_this(io);
        use_this.set(type_all());
        read_all(use_this);
        io.reference(use_this.get<T,N>());
      }

      template <typename T, int N> blitz::Array<T,N> read_all() {
        bob::core::array::blitz_array tmp(type_all());
        read_all(tmp);
        return tmp.get<T,N>();
      }

      template <typename T, int N> size_t append(const blitz::Array<T,N>& in) {
        bob::core::array::blitz_array use_this(in);
        return append(use_this);
      }

      template <typename T, int N> void write (const blitz::Array<T,N>& in) {
        bob::core::array::blitz_array use_this(in);
        write(use_this);
      }

  };

  /**
   * @brief This defines the factory method F that can create codecs. Your 
   * task, as a codec developer is to create one of such methods for each of 
   * your codecs and statically register them to the codec registry.
   *
   * Here are the meanings of the mode flag that should be respected by your
   * factory implementation:
   *
   * 'r': opens for reading only - no modifications can occur; it is an
   *      error to open a file that does not exist for read-only operations.
   * 'w': opens for reading and writing, but truncates the file if it
   *      exists; it is not an error to open files that do not exist with
   *      this flag. 
   * 'a': opens for reading and writing - any type of modification can 
   *      occur. If the file does not exist, this flag is effectively like
   *      'w'.
   *
   * Returns a newly allocated File object that can read and write data to the
   * file using a specific backend.
   */
  typedef boost::shared_ptr<File> (*file_factory_t)
    (const std::string& filename, char mode);

  /**
   * @}
   */
}}

#endif /* BOB_IO_FILE_H */
