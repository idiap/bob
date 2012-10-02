/**
 * @file io/cxx/HDF5Attribute.cc
 * @author Andre Anjos <andre.anjos@idiap.ch>
 * @date Fri  2 Mar 08:23:47 2012
 *
 * @brief Implements attribute read/write for HDF5 files
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

#include <boost/format.hpp>
#include "bob/io/HDF5Attribute.h"
#include "bob/io/HDF5Exception.h"
#include "bob/core/logging.h"

namespace h5 = bob::io::detail::hdf5;
namespace io = bob::io;

bool h5::has_attribute(const boost::shared_ptr<hid_t> location,
    const std::string& name) {
  return H5Aexists(*location, name.c_str());
}

/**
 * Opens an "auto-destructible" HDF5 dataspace
 */
static void delete_h5dataspace (hid_t* p) {
  if (*p >= 0) {
    herr_t err = H5Sclose(*p);
    if (err < 0) {
      bob::core::error << "H5Sclose() exited with an error (" << err << "). The stack trace follows:" << std::endl;
      bob::core::error << bob::io::format_hdf5_error() << std::endl;
    }
  }
  delete p;
}

static boost::shared_ptr<hid_t> open_memspace(const io::HDF5Type& t) {
  boost::shared_ptr<hid_t> retval(new hid_t(-1), std::ptr_fun(delete_h5dataspace));
  *retval = H5Screate_simple(t.shape().n(), t.shape().get(), 0);
  if (*retval < 0) throw io::HDF5StatusError("H5Screate_simple", *retval);
  return retval;
}

/**
 * Opens the memory space of attribute
 */
static boost::shared_ptr<hid_t> get_memspace(hid_t attr) {
  boost::shared_ptr<hid_t> retval(new hid_t(-1), std::ptr_fun(delete_h5dataspace));
  *retval = H5Aget_space(attr);
  if (*retval < 0) throw io::HDF5StatusError("H5Aget_space", *retval);
  return retval;
}

/**
 * Auto-destructing HDF5 type
 */
static void delete_h5type (hid_t* p) {
  if (*p >= 0) {
    herr_t err = H5Tclose(*p);
    if (err < 0) {
      bob::core::error << "H5Tclose() exited with an error (" << err << "). The stack trace follows:" << std::endl;
      bob::core::error << bob::io::format_hdf5_error() << std::endl;
    }
  }
  delete p;
}

/**
 * Gets datatype of attribute
 */
static boost::shared_ptr<hid_t> get_type(hid_t attr) {
  boost::shared_ptr<hid_t> retval(new hid_t(-1), std::ptr_fun(delete_h5type));
  *retval = H5Aget_type(attr);
  if (*retval < 0) throw io::HDF5StatusError("H5Aget_type", *retval);
  return retval;
}

/**
 * Figures out the extents of an attribute
 */
static io::HDF5Shape get_extents(hid_t space) {
  int rank = H5Sget_simple_extent_ndims(space);
  if (rank < 0) throw io::HDF5StatusError("H5Sget_simple_extent_ndims", rank);
  //is at least a list of scalars, but could be a list of arrays
  io::HDF5Shape shape(rank);
  herr_t status = H5Sget_simple_extent_dims(space, shape.get(), 0);
  if (status < 0) throw io::HDF5StatusError("H5Sget_simple_extent_dims",status);
  return shape;
}

/**
 * Opens an "auto-destructible" HDF5 attribute
 */
static void delete_h5attribute (hid_t* p) {
  if (*p >= 0) {
    herr_t err = H5Aclose(*p);
    if (err < 0) {
      bob::core::error << "H5Aclose() exited with an error (" << err << "). The stack trace follows:" << std::endl;
      bob::core::error << bob::io::format_hdf5_error() << std::endl;
    }
  }
  delete p;
}

static boost::shared_ptr<hid_t> open_attribute
(const boost::shared_ptr<hid_t> location, const std::string& name,
 const io::HDF5Type& t) {

  boost::shared_ptr<hid_t> retval(new hid_t(-1),
      std::ptr_fun(delete_h5attribute));

  *retval = H5Aopen(*location, name.c_str(), H5P_DEFAULT);

  if (*retval < 0) throw io::HDF5StatusError("H5Aopen", *retval);

  //checks if the opened attribute is compatible w/ the expected type
  boost::shared_ptr<hid_t> atype = get_type(*retval);
  boost::shared_ptr<hid_t> aspace = get_memspace(*retval);
  bob::io::HDF5Shape shape = get_extents(*aspace);
  io::HDF5Type expected(atype, shape);
  if (expected != t) {
    boost::format m("Trying to access attribute '%s' with incompatible buffer - expected `%s', but you gave me `%s'");
    m % name % expected.str() % t.str();
    throw std::runtime_error(m.str());
  }

  return retval;
}

void h5::delete_attribute (boost::shared_ptr<hid_t> location,
    const std::string& name) {
  herr_t err = H5Adelete(*location, name.c_str());
  if (err < 0) throw io::HDF5StatusError("H5Adelete", err);
}

void h5::read_attribute (const boost::shared_ptr<hid_t> location,
    const std::string& name, const bob::io::HDF5Type& dest,
    void* buffer) {
  boost::shared_ptr<hid_t> attribute = open_attribute(location, name, dest);
  herr_t err = H5Aread(*attribute, *dest.htype(), buffer);
  if (err < 0) throw io::HDF5StatusError("H5Aread", err);
}

void h5::gettype_attribute (const boost::shared_ptr<hid_t> location,
    const std::string& name, io::HDF5Type& type) {

  boost::shared_ptr<hid_t> attr(new hid_t(-1),
      std::ptr_fun(delete_h5attribute));

  *attr = H5Aopen(*location, name.c_str(), H5P_DEFAULT);

  if (*attr < 0) throw io::HDF5StatusError("H5Aopen", *attr);

  boost::shared_ptr<hid_t> atype = get_type(*attr);
  boost::shared_ptr<hid_t> aspace = get_memspace(*attr);
  bob::io::HDF5Shape shape = get_extents(*aspace);

  type = bob::io::HDF5Type(atype, shape);
}

static boost::shared_ptr<hid_t> create_attribute(boost::shared_ptr<hid_t> loc,
    const std::string& name, const io::HDF5Type& t,
    boost::shared_ptr<hid_t> space) {

  boost::shared_ptr<hid_t> retval(new hid_t(-1),
      std::ptr_fun(delete_h5attribute));

  *retval = H5Acreate2(*loc, name.c_str(), *t.htype(), *space, H5P_DEFAULT,
      H5P_DEFAULT);

  if (*retval < 0) throw io::HDF5StatusError("H5Acreate", *retval);
  return retval;
}

void h5::write_attribute (boost::shared_ptr<hid_t> location,
    const std::string& name, const bob::io::HDF5Type& dest, const void* buffer)
{

  boost::shared_ptr<hid_t> dataspace = open_memspace(dest);

  if (h5::has_attribute(location, name)) h5::delete_attribute(location, name);
  boost::shared_ptr<hid_t> attribute =
    create_attribute(location, name, dest, dataspace);

  /* Write the attribute data. */
  herr_t err = H5Awrite(*attribute, *dest.htype(), buffer);
  if (err < 0) throw io::HDF5StatusError("H5Awrite", err);
}

static herr_t attr_iterator (hid_t attr, const char* name, const H5A_info_t*, 
    void* cookie) {
  std::map<std::string, io::HDF5Type>& dict =
    *static_cast<std::map<std::string, io::HDF5Type>*>(cookie);
  boost::shared_ptr<hid_t> atype = get_type(attr);
  boost::shared_ptr<hid_t> aspace = get_memspace(attr);
  bob::io::HDF5Shape shape = get_extents(*aspace);
  dict[name] = bob::io::HDF5Type(atype, shape);
  return 0;
}

void h5::list_attributes(boost::shared_ptr<hid_t> location,
    std::map<std::string, io::HDF5Type>& attributes) {
  hsize_t offset=0;
  H5Aiterate2(*location, H5_INDEX_NAME, H5_ITER_NATIVE, &offset, attr_iterator, 
      static_cast<void*>(&attributes));
}
