/**
 * @file bob/io/HDF5Dataset.h
 * @author Andre Anjos <andre.anjos@idiap.ch>
 * @date Wed 29 Feb 17:27:45 2012
 *
 * @brief Describes HDF5 datasets
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

#ifndef BOB_IO_HDF5DATASET_H
#define BOB_IO_HDF5DATASET_H

#include <vector>

#include <boost/shared_ptr.hpp>
#include <blitz/array.h>
#if !defined (HAVE_BLITZ_TINYVEC2_H)
#include <blitz/tinyvec-et.h>
#endif
#include <hdf5.h>

#include "bob/core/Exception.h"
#include "bob/core/array_assert.h"
#include "bob/core/array_copy.h"

#include "bob/io/HDF5Attribute.h"
#include "bob/io/HDF5Exception.h"
#include "bob/io/HDF5Types.h"

namespace bob { namespace io { namespace detail { namespace hdf5 {

  class File;
  class Group;

  /**
   * An HDF5 C-style dataset that knows how to close itself.
   */
  class Dataset {

    public: //better to protect?

      /**
       * Creates a new HDF5 dataset by reading its contents from a certain
       * file.
       */
      Dataset(boost::shared_ptr<Group> parent, const std::string& name);

      /**
       * Creates a new HDF5 dataset from scratch and inserts it in the given
       * group. If the Dataset already exists on file and the types are
       * compatible, we attach to that type, otherwise, we raise an exception.
       *
       * If a new Dataset is to be created, you can also set if you would like
       * to have as a list and the compression level. Note these settings have
       * no effect if the Dataset already exists on file, in which case the
       * current settings for that dataset are respected. The maximum value for
       * the gzip compression is 9. The value of zero turns compression off
       * (the default).
       *
       * The effect of setting "list" to false is that the created dataset:
       *
       * a) Will not be expandable (chunked)
       * b) Will contain the exact number of dimensions of the input type.
       *
       * When you set "list" to true (the default), datasets are created with
       * chunking automatically enabled (the chunk size is set to the size of
       * the given variable) and an extra dimension is inserted to accommodate
       * list operations.
       */
      Dataset(boost::shared_ptr<Group> parent, const std::string& name,
          const bob::io::HDF5Type& type, bool list=true,
          size_t compression=0);

    public: //api

      /**
       * Destructor virtualization
       */
      virtual ~Dataset();

      /**
       * Returns the number of objects installed at this dataset from the
       * perspective of the default compatible type.
       */
      size_t size() const;

      /**
       * Returns the number of objects installed at this dataset from the
       * perspective of the default compatible type. If the given type is not
       * compatible, raises a type error.
       */
      size_t size(const bob::io::HDF5Type& type) const;

      /**
       * Get parent group
       */
      virtual const boost::shared_ptr<Group> parent() const;
      virtual boost::shared_ptr<Group> parent();

      /**
       * Filename where I'm sitting
       */
      virtual const std::string& filename() const;

      /**
       * Full path to myself. Constructed each time it is called.
       */
      virtual std::string path() const;

      /**
       * Path with filename. Constructed each time it is called.
       */
      virtual std::string url() const;

      /**
       * Access file
       */
      virtual const boost::shared_ptr<File> file() const;
      virtual boost::shared_ptr<File> file();

      /**
       * My name
       */
      virtual const std::string& name() const {
        return m_name;
      }

      /**
       * Accesses the current location id of this dataset
       */
      const boost::shared_ptr<hid_t> location() const {
        return m_id;
      }
      boost::shared_ptr<hid_t> location() {
        return m_id;
      }

      /**
       * DATA READING FUNCTIONALITY
       */

      /**
       * Reads data from the file into a scalar. The conditions bellow have to
       * be respected:
       *
       * a. My internal shape is 1D **OR** my internal shape is 2D, but the
       *    extent of the second dimension is 1.
       * b. The indexed position exists
       *
       * If the internal shape is not like defined above, raises a type error.
       * If the indexed position does not exist, raises an index error.
       */
      template <typename T> void read(size_t index, T& value) {
        bob::io::HDF5Type dest_type(value);
        read_buffer(index, dest_type, reinterpret_cast<void*>(&value));
      }

      /**
       * Reads data from the file into a scalar (allocated internally). The
       * same conditions as for read(index, value) apply.
       */
      template <typename T> T read(size_t index) {
        T retval;
        read(index, retval);
        return retval;
      }

      /**
       * Reads data from the file into a scalar. This is equivalent to using
       * read(0). The same conditions as for read(index=0, value) apply.
       */
      template <typename T> T read() {
        T retval;
        read(0, retval);
        return retval;
      }

      /**
       * Reads data from the file into a array. The following conditions have
       * to be respected:
       *
       * a. My internal shape is the same as the shape of the given value
       *    **OR** my internal shape has one more dimension as the given value.
       *    In this case, the first dimension of the internal shape is
       *    considered to be an index and the remaining shape values the
       *    dimension of the value to be read. The given array has to be
       *    compatible with this re-defined N-1 shape.
       * b. The indexed position exists
       *
       * If the internal shape is not like defined above, raises a type error.
       * If the index does not exist, raises an index error.
       *
       * @param index Which of the arrays to read in the current dataset, by
       * order
       * @param value The output array data will be stored inside this
       * variable. This variable has to be a zero-based C-style contiguous
       * storage array. If that is not the case, we will raise an exception.
       */
      template <typename T, int N>
        void readArray(size_t index, blitz::Array<T,N>& value) {
          bob::core::array::assertCZeroBaseContiguous(value);
          bob::io::HDF5Type dest_type(value);
          read_buffer(index, dest_type, reinterpret_cast<void*>(value.data()));
        }

      /**
       * Reads data from the file into an array allocated dynamically. The same
       * conditions as for readArray(index, value) apply.
       *
       * @param index Which of the arrays to read in the current dataset, by
       * order
       */
      template <typename T, int N>
        blitz::Array<T,N> readArray(size_t index) {
          for (size_t k=m_descr.size(); k>0; --k) {
            const bob::io::HDF5Shape& S = m_descr[k-1].type.shape();
            if(S.n() == N) {
              blitz::TinyVector<int,N> shape;
              S.set(shape);
              blitz::Array<T,N> retval(shape);
              readArray(index, retval);
              return retval;
            }
          }
          throw bob::io::HDF5IncompatibleIO(url(),
              m_descr[0].type.str(), "dynamic shape unknown");
        }

      /**
       * Reads data from the file into a array. This is equivalent to using
       * readArray(0, value). The same conditions as for readArray(index=0,
       * value) apply.
       *
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
       * readArray(0). The same conditions as for readArray(index=0, value)
       * apply.
       */
      template <typename T, int N>
        blitz::Array<T,N> readArray() {
          return readArray<T,N>(0);
        }

      /**
       * DATA WRITING FUNCTIONALITY
       */

      /**
       * Modifies the value of a scalar inside the file. Modifying a value
       * requires that the expected internal shape for this dataset and the
       * shape of the given scalar are consistent. To replace a scalar the
       * conditions bellow have to be respected:
       *
       * a. The internal shape is 1D **OR** the internal shape is 2D, but the
       *    second dimension of the internal shape has is extent == 1.
       * b. The given indexing position exists
       *
       * If the above conditions are not met, an exception is raised.
       */
      template <typename T> void replace(size_t index, const T& value) {
        bob::io::HDF5Type dest_type(value);
        write_buffer(index, dest_type, reinterpret_cast<const void*>(&value));
      }

      /**
       * Modifies the value of a scalar inside the file. This is equivalent to
       * using replace(0, value). The same conditions as for replace(index=0,
       * value) apply.
       */
      template <typename T> void replace(const T& value) {
        replace(0, value);
      }

      /**
       * Inserts a scalar in the current (existing ;-) dataset. This will
       * trigger writing data to the file. Adding a scalar value requires that
       * the expected internal shape for this dataset and the shape of the
       * given scalar are consistent. To add a scalar the conditions
       * bellow have to be respected:
       *
       * a. The internal shape is 1D **OR** the internal shape is 2D, but the
       *    second dimension of the internal shape has is extent == 1.
       * b. This dataset is expandible (chunked)
       *
       * If the above conditions are not met, an exception is raised.
       */
      template <typename T> void add(const T& value) {
        bob::io::HDF5Type dest_type(value);
        extend_buffer(dest_type, reinterpret_cast<const void*>(&value));
      }

      /**
       * Replaces data at the file using a new array. Replacing an existing
       * array requires shape consistence. The following conditions should be
       * met:
       *
       * a. My internal shape is the same as the shape of the given value
       *    **OR** my internal shape has one more dimension as the given value.
       *    In this case, the first dimension of the internal shape is
       *    considered to be an index and the remaining shape values the
       *    dimension of the value to be read. The given array has to be
       *    compatible with this re-defined N-1 shape.
       * b. The given indexing position exists.
       *
       * If the internal shape is not like defined above, raises a type error.
       * If the indexed position does not exist, raises an index error.
       *
       * @param index Which of the arrays to read in the current dataset, by
       * order
       * @param value The output array data will be stored inside this
       * variable. This variable has to be a zero-based C-style contiguous
       * storage array. If that is not the case, we will raise an exception.
       */
      template <typename T, int N>
        void replaceArray(size_t index, const blitz::Array<T,N>& value) {
          bob::io::HDF5Type dest_type(value);
          if(!bob::core::array::isCZeroBaseContiguous(value)) {
            blitz::Array<T,N> tmp = bob::core::array::ccopy(value);
            write_buffer(index, dest_type, reinterpret_cast<const void*>(tmp.data()));
          }
          else {
            write_buffer(index, dest_type,
                reinterpret_cast<const void*>(value.data()));
          }
        }

      /**
       * Replaces data at the file using a new array. This is equivalent to
       * calling replaceArray(0, value). The conditions for
       * replaceArray(index=0, value) apply.
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
       * subdirectory (or a "group" in HDF5 parlance) does not exist, it is
       * created. If the dataset does not exist, it is created, otherwise, we
       * append to it. In this case, the dimensionality of the scalar has to be
       * compatible with the existing dataset shape (or "dataspace" in HDF5
       * parlance). If you want to do this, first unlink and than use one of
       * the add() methods.
       */
      template <typename T, int N>
        void addArray(const blitz::Array<T,N>& value) {
          bob::io::HDF5Type dest_type(value);
          if(!bob::core::array::isCZeroBaseContiguous(value)) {
            blitz::Array<T,N> tmp = bob::core::array::ccopy(value);
            extend_buffer(dest_type, reinterpret_cast<const void*>(tmp.data()));
          }
          else {
            extend_buffer(dest_type, reinterpret_cast<const void*>(value.data()));
          }
      }

    private: //apis

      /**
       * Selects a bit of the file to be affected at the next read or write
       * operation. This method encapsulate calls to H5Sselect_hyperslab().
       *
       * The index is checked for existence as well as the consistence of the
       * destination type.
       */
      std::vector<bob::io::HDF5Descriptor>::iterator select (size_t index,
          const bob::io::HDF5Type& dest);

    public: //direct access for other bindings -- don't use these!

      /**
       * Reads a previously selected area into the given (user) buffer.
       */
      void read_buffer (size_t index, const bob::io::HDF5Type& dest, void* buffer);

      /**
       * Writes the contents of a given buffer into the file. The area that the
       * data will occupy should have been selected beforehand.
       */
      void write_buffer (size_t index, const bob::io::HDF5Type& dest,
          const void* buffer);

      /**
       * Extend the dataset with one extra variable.
       */
      void extend_buffer (const bob::io::HDF5Type& dest, const void* buffer);

    public: //attribute support

      /**
       * Gets the current type set for an attribute
       */
      void gettype_attribute(const std::string& name,
          bob::core::array::typeinfo& type) const;

      /**
       * Sets a scalar attribute on the current group. Setting an existing
       * attribute overwrites its value.
       */
      template <typename T> void set_attribute(const std::string& name,
          const T& v) {
        bob::io::HDF5Type dest_type(v);
        write_attribute(m_id,name,dest_type,reinterpret_cast<const void*>(&v));
      }

      /**
       * Reads an attribute from the current dataset. Raises an error if such
       * attribute does not exist on the group. To check for existence, use
       * has_attribute().
       */
      template <typename T> T get_attribute(const std::string& name) const {
        T v;
        bob::io::HDF5Type dest_type(v);
        read_attribute(m_id, name, dest_type, reinterpret_cast<void*>(&v));
        return v;
      }

      /**
       * Checks if a certain attribute exists in this group.
       */
      bool has_attribute(const std::string& name) const;

      /**
       * Deletes an attribute
       */
      void delete_attribute(const std::string& name);

    public: //array attribute support

      /**
       * Sets a array attribute on the current group. Setting an existing
       * attribute overwrites its value. If the attribute exists it is erased
       * and re-written.
       */
      template <typename T, int N> void set_array_attribute(const std::string& name,
          const blitz::Array<T,N>& v) {
        bob::io::HDF5Type dest_type(v);
        if(!bob::core::array::isCZeroBaseContiguous(v)) {
          blitz::Array<T,N> tmp = bob::core::array::ccopy(v);
          write_attribute(name, dest_type, reinterpret_cast<const void*>(tmp.data()));
        }
        else {
          write_attribute(name, dest_type, reinterpret_cast<const void*>(v.data()));
        }
      }

      /**
       * Reads an attribute from the current dataset. Raises an error if such
       * attribute does not exist on the group. To check for existence, use
       * has_attribute().
       */
      template <typename T,N> blitz::Array<T,N> get_array_attribute(const std::string& name) const {
        blitz::Array<T,N> v;
        bob::io::HDF5Type dest_type(v);
        read_attribute(name, dest_type, reinterpret_cast<void*>(v.data()));
        return v;
      }

      /**
       * Reads an attribute from the current dataset. Places the data in an
       * already allocated array.
       */
      template <typename T,N> void get_array_attribute(const std::string& name,
          blitz::Array<T,N>& v) const {
        bob::io::HDF5Type dest_type(v);
        read_attribute(name, dest_type, reinterpret_cast<void*>(v.data()));
      }

    public: //buffer attribute support

      /**
       * Reads an attribute into a user buffer. It is the user's responsibility
       * to have a buffer that represents the given type.
       */
      void read_attribute (const std::string& name, 
          const bob::io::HDF5Type& dest, void* buffer) const;

      /**
       * Writes the contents of a given buffer into the attribute.
       */
      void write_attribute (const std::string& name,
          const bob::io::HDF5Type& dest, const void* buffer);

    private: //not implemented

      /**
       * Copies the contents of an existing dataset -- not implemented
       */
      Dataset(const Dataset& other);

      /**
       * Assigns the contents of an existing dataset to myself -- not
       * implemented
       */
      Dataset& operator= (const Dataset& other);

    public: //representation

      boost::weak_ptr<Group> m_parent; ///< my parent group
      std::string m_name; ///< name of this object
      boost::shared_ptr<hid_t> m_id; ///< the HDF5 Dataset this type points to
      boost::shared_ptr<hid_t> m_dt; ///< the datatype of this Dataset
      boost::shared_ptr<hid_t> m_filespace; ///< the "file" space for this set
      std::vector<bob::io::HDF5Descriptor> m_descr; ///< read/write descr.'s
      boost::shared_ptr<hid_t> m_memspace; ///< read/write space

  };

}}}}


#endif /* BOB_IO_HDF5DATASET_H */
