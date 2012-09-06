/**
 * @file bob/io/HDF5Attribute.h
 * @author Andre Anjos <andre.anjos@idiap.ch>
 * @date Fri  2 Mar 08:19:03 2012 
 *
 * @brief Simple attribute support for HDF5 files
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

#ifndef BOB_IO_HDF5ATTRIBUTE_H 
#define BOB_IO_HDF5ATTRIBUTE_H

#include <string>
#include <boost/shared_ptr.hpp>
#include <hdf5.h>
#include "bob/io/HDF5Types.h"

namespace bob { namespace io { namespace detail { namespace hdf5 {

  /**
   * reads the attribute value, place it in "buffer"
   */
  void read_attribute (const boost::shared_ptr<hid_t> location,
      const std::string& name, const bob::io::HDF5Type& dest, void* buffer);

  /**
   * writes an attribute value from "buffer"
   */
  void write_attribute (boost::shared_ptr<hid_t> location,
      const std::string& name, const bob::io::HDF5Type& dest,
      const void* buffer);

  /**
   * Sets a scalar attribute on the given location. Setting an existing
   * attribute overwrites its value.
   *
   * @note Only simple scalars are supported for the time being
   */
  template <typename T> void set_attribute(boost::shared_ptr<hid_t> location,
      const std::string& name, const T& v) {
    bob::io::HDF5Type dest_type(v);
    write_attribute(location, name, dest_type, 
        reinterpret_cast<const void*>(&v));
  }

  /**
   * Reads an attribute from the current group. Raises an error if such
   * attribute does not exist on the group. To check for existence, use
   * has_attribute().
   */
  template <typename T> T get_attribute(const boost::shared_ptr<hid_t> location,
      const std::string& name) {
    T v;
    bob::io::HDF5Type dest_type(v);
    read_attribute(location, name, dest_type, reinterpret_cast<void*>(&v));
    return v;
  }

  /**
   * Checks if a certain attribute exists in this location.
   */
  bool has_attribute(const boost::shared_ptr<hid_t> location, 
      const std::string& name);

  /**
   * Deletes an attribute from a location.
   */
  void delete_attribute(boost::shared_ptr<hid_t> location,
      const std::string& name);

}}}}

#endif /* BOB_IO_HDF5ATTRIBUTE_H */
