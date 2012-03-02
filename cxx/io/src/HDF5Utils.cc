/**
 * @file cxx/io/src/HDF5Utils.cc
 * @date Wed Jun 22 17:50:08 2011 +0200
 * @author Andre Anjos <andre.anjos@idiap.ch>
 *
 * @brief Implements a set of utilities to read HDF5 files.
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

#include <boost/make_shared.hpp>
#include "io/HDF5Utils.h"

namespace h5 = bob::io::detail::hdf5;
namespace io = bob::io;

/**
 * Opens/Creates an "auto-destructible" HDF5 file
 */
static void delete_h5file (hid_t* p) {
  if (*p >= 0) {
    H5Fclose(*p);
  }
  delete p;
  p=0;
}

/**
 * Opens/Creates and "auto-destructible" HDF5 file creation property list
 */
static void delete_h5p (hid_t* p) {
  if (*p >= 0) {
    H5Pclose(*p);
  }
  delete p;
  p=0;
}

static boost::shared_ptr<hid_t> open_file(const boost::filesystem::path& path,
    unsigned int flags, boost::shared_ptr<hid_t>& fcpl) {

  boost::shared_ptr<hid_t> retval(new hid_t(-1), std::ptr_fun(delete_h5file));

  if (!boost::filesystem::exists(path) && flags == H5F_ACC_RDONLY) {
    //file was opened for reading, but does not exist... Raise
    throw io::FileNotReadable(path.string());
  }

  if (boost::filesystem::exists(path) && flags != H5F_ACC_TRUNC) { //open
    *retval = H5Fopen(path.string().c_str(), flags, H5P_DEFAULT);
    if (*retval < 0) throw io::HDF5StatusError("H5Fopen", *retval);
    //replaces the file create list properties with the one from the file
    fcpl = boost::shared_ptr<hid_t>(new hid_t(-1), std::ptr_fun(delete_h5p));
    *fcpl = H5Fget_create_plist(*retval);
    if (*fcpl < 0) throw io::HDF5StatusError("H5Fget_create_list", *fcpl);
  }
  else { //file needs to be created or truncated (can set user block)
    *retval = H5Fcreate(path.string().c_str(), H5F_ACC_TRUNC,
        *fcpl, H5P_DEFAULT);
    if (*retval < 0) throw io::HDF5StatusError("H5Fcreate", *retval);
  }

  return retval;
}

static boost::shared_ptr<hid_t> create_fcpl(hsize_t userblock_size) {
  if (!userblock_size) return boost::make_shared<hid_t>(H5P_DEFAULT);
  //otherwise we have to go through the settings
  boost::shared_ptr<hid_t> retval(new hid_t(-1), std::ptr_fun(delete_h5p));
  *retval = H5Pcreate(H5P_FILE_CREATE);
  if (*retval < 0) throw io::HDF5StatusError("H5Pcreate", *retval);
  herr_t err = H5Pset_userblock(*retval, userblock_size);
  if (err < 0) throw io::HDF5StatusError("H5Pset_userblock", err);
  return retval;
}

h5::File::File(const boost::filesystem::path& path, unsigned int flags,
    size_t userblock_size):
  m_path(path),
  m_flags(flags),
  m_fcpl(create_fcpl(userblock_size)),
  m_id(open_file(m_path, m_flags, m_fcpl))
{
}

h5::File::~File() {
}

boost::shared_ptr<h5::RootGroup> h5::File::root() {
  if (!m_root) {
    m_root = boost::make_shared<h5::RootGroup>(shared_from_this());
    m_root->open_recursively();
  }
  return m_root;
}

void h5::File::reset() {
  m_root.reset();
}

size_t h5::File::userblock_size() const {
  hsize_t retval;
  herr_t err = H5Pget_userblock(*m_fcpl, &retval);
  if (err < 0) throw io::HDF5StatusError("H5Pget_create_plist", err);
  return retval;
}

void h5::File::get_userblock(std::string& data) const {
  //TODO
}

void h5::File::set_userblock(const std::string& data) {
  //TODO
}
