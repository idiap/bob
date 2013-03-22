/**
 * @file io/cxx/HDF5Group.cc
 * @author Andre Anjos <andre.anjos@idiap.ch>
 * @date Wed 29 Feb 17:24:10 2012
 *
 * @brief Implements HDF5 groups.
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

#include <boost/make_shared.hpp>
#include <boost/shared_array.hpp>
#include <boost/filesystem.hpp>
#include <boost/format.hpp>
#include <boost/algorithm/string.hpp>
#include <bob/io/HDF5Group.h>
#include <bob/io/HDF5Utils.h>
#include <bob/core/logging.h>

/**
 * Creates an "auto-destructible" HDF5 Group
 */
static void delete_h5g (hid_t* p) {
  if (*p >= 0) {
    herr_t err = H5Gclose(*p);
    if (err < 0) {
      bob::core::error << "H5Gclose() exited with an error (" << err << "). The stack trace follows:" << std::endl;
      bob::core::error << bob::io::format_hdf5_error() << std::endl;
    }
  }
  delete p;
}

static boost::shared_ptr<hid_t> create_new_group(boost::shared_ptr<hid_t> p,
    const std::string& name) {
  boost::shared_ptr<hid_t> retval(new hid_t(-1), std::ptr_fun(delete_h5g));
  *retval = H5Gcreate2(*p, name.c_str(), H5P_DEFAULT, H5P_DEFAULT,
      H5P_DEFAULT);
  if (*retval < 0) throw bob::io::HDF5StatusError("H5Gcreate", *retval);
  return retval;
}

static boost::shared_ptr<hid_t> open_group(boost::shared_ptr<hid_t> g,
    const char* name) {
  boost::shared_ptr<hid_t> retval(new hid_t(-1), std::ptr_fun(delete_h5g));
  *retval = H5Gopen2(*g, name, H5P_DEFAULT);
  if (*retval < 0) throw bob::io::HDF5StatusError("H5Gopen", *retval);
  return retval;
}

bob::io::detail::hdf5::Group::Group(boost::shared_ptr<Group> parent, const std::string& name):
  m_name(name),
  m_id(create_new_group(parent->location(), name)),
  m_parent(parent)
{
}

/**
 * Simple wrapper to call internal bob::io::detail::hdf5::Group::iterate_callback, that can call
 * Group and Dataset constructors. Note that those are private or protected for
 * design reasons.
 */
static herr_t group_iterate_callback(hid_t self, const char *name,
    const H5L_info_t *info, void *object) {
  return static_cast<bob::io::detail::hdf5::Group*>(object)->iterate_callback(self, name, info);
}

herr_t bob::io::detail::hdf5::Group::iterate_callback(hid_t self, const char *name,
    const H5L_info_t *info) {

  // If we are not looking at a hard link to the data, just ignore
  if (info->type != H5L_TYPE_HARD) {
    TDEBUG1("Ignoring soft-link `" << name << "' in HDF5 file");
    return 0;
  }

  // Get information about the HDF5 object
  H5O_info_t obj_info;
  herr_t status = H5Oget_info_by_name(self, name, &obj_info, H5P_DEFAULT);
  if (status < 0) throw bob::io::HDF5StatusError("H5Oget_info_by_name", status);

  switch(obj_info.type) {
    case H5O_TYPE_GROUP:
      //creates with recursion
      m_groups[name] = boost::make_shared<bob::io::detail::hdf5::Group>(shared_from_this(),
          name, true);
      m_groups[name]->open_recursively();
      break;
    case H5O_TYPE_DATASET:
      m_datasets[name] = boost::make_shared<bob::io::detail::hdf5::Dataset>(shared_from_this(),
          std::string(name));
      break;
    default:
      break;
  }

  return 0;
}

bob::io::detail::hdf5::Group::Group(boost::shared_ptr<Group> parent,
    const std::string& name, bool):
  m_name(name),
  m_id(open_group(parent->location(), name.c_str())),
  m_parent(parent)
{
  //checks name
  if (!m_name.size() || m_name == "." || m_name == "..") {
    boost::format m("Cannot create group with illegal name `%s' at `%s'");
    m % name % url();
    throw std::runtime_error(m.str());
  }
}

void bob::io::detail::hdf5::Group::open_recursively() {
  //iterates over this group only and instantiates what needs to be instantiated
  herr_t status = H5Literate(*m_id, H5_INDEX_NAME,
      H5_ITER_NATIVE, 0, group_iterate_callback, static_cast<void*>(this));
  if (status < 0) throw bob::io::HDF5StatusError("H5Literate", status);
}

bob::io::detail::hdf5::Group::Group(boost::shared_ptr<File> parent):
  m_name(""),
  m_id(open_group(parent->location(), "/")),
  m_parent()
{
}

bob::io::detail::hdf5::Group::~Group() { }

const boost::shared_ptr<bob::io::detail::hdf5::Group> bob::io::detail::hdf5::Group::parent() const {
  return m_parent.lock();
}

boost::shared_ptr<bob::io::detail::hdf5::Group> bob::io::detail::hdf5::Group::parent() {
  return m_parent.lock();
}

const std::string& bob::io::detail::hdf5::Group::filename() const {
  return parent()->filename();
}

std::string bob::io::detail::hdf5::Group::path() const {
  return (m_name.size()?parent()->path():"") + "/" + m_name;
}

std::string bob::io::detail::hdf5::Group::url() const {
  return filename() + ":" + path();
}

const boost::shared_ptr<bob::io::detail::hdf5::File> bob::io::detail::hdf5::Group::file() const {
  return parent()->file();
}

boost::shared_ptr<bob::io::detail::hdf5::File> bob::io::detail::hdf5::Group::file() {
  return parent()->file();
}

boost::shared_ptr<bob::io::detail::hdf5::Group> bob::io::detail::hdf5::Group::cd(const std::string& dir) {
  //empty dir == void action, return self
  if (!dir.size()) return shared_from_this();

  if (dir[0] == '/') { //absolute path given, apply to root node
    return file()->root()->cd(dir.substr(1));
  }

  //relative path given, start from self
  std::string::size_type pos = dir.find_first_of('/');
  if (pos == std::string::npos) { //it should be one of my children
    if (dir == ".") return shared_from_this();
    if (dir == "..") {
      if (!m_name.size()) { //this is the root group already
        boost::format m("Cannot go beyond root directory at file `%s'");
        m % file()->filename();
        throw std::runtime_error(m.str());
      }
      //else, just return its parent
      return parent();
    }
    if (!has_group(dir)) {
      boost::format m("Cannot find group `%s' at `%s'");
      m % dir % url();
      throw std::runtime_error(m.str());
    }
    //else, just return the named group
    return m_groups[dir];
  }

  //if you get to this point, we are just traversing
  std::string mydir = dir.substr(0, pos);
  if (mydir == ".") return cd(dir.substr(pos+1));
  if (mydir == "..") return parent()->cd(dir.substr(pos+1));
  if (!has_group(mydir)) {
    boost::format m("Cannot find group `%s' at `%s'");
    m % dir % url();
    throw std::runtime_error(m.str());
  }

  //else, just recurse to the next group
  return m_groups[mydir]->cd(dir.substr(pos+1));
}

const boost::shared_ptr<bob::io::detail::hdf5::Group> bob::io::detail::hdf5::Group::cd(const std::string& dir) const {
  return const_cast<bob::io::detail::hdf5::Group*>(this)->cd(dir);
}

boost::shared_ptr<bob::io::detail::hdf5::Dataset> bob::io::detail::hdf5::Group::operator[] (const std::string& dir) {
  std::string::size_type pos = dir.find_last_of('/');
  if (pos == std::string::npos) { //search on the current group
    if (!has_dataset(dir)) {
      boost::format m("Cannot find dataset `%s' at `%s'");
      m % dir % url();
      throw std::runtime_error(m.str());
    }
    return m_datasets[dir];
  }

  //if you get to this point, the search routine needs to be performed on
  //another group, indicated by the path. So, we first cd() there and then do
  //the same as we do here. This will recurse through the directory structure
  //until we find the place defined by the user or raise an exception.
  std::string dest = dir.substr(0, pos);
  if (!dest.size()) dest = "/";
  boost::shared_ptr<bob::io::detail::hdf5::Group> g = cd(dest);
  return g->operator[](dir.substr(pos+1));
}

const boost::shared_ptr<bob::io::detail::hdf5::Dataset> bob::io::detail::hdf5::Group::operator[] (const std::string& dir) const {
  return const_cast<bob::io::detail::hdf5::Group*>(this)->operator[](dir);
}

void bob::io::detail::hdf5::Group::reset() {
  typedef std::map<std::string, boost::shared_ptr<bob::io::detail::hdf5::Group> > group_map_type;
  for (group_map_type::const_iterator it = m_groups.begin();
      it != m_groups.end(); ++it) {
    remove_group(it->first);
  }

  typedef std::map<std::string, boost::shared_ptr<bob::io::detail::hdf5::Dataset> >
    dataset_map_type;
  for (dataset_map_type::const_iterator it = m_datasets.begin();
      it != m_datasets.end(); ++it) {
    remove_dataset(it->first);
  }
}

boost::shared_ptr<bob::io::detail::hdf5::Group> bob::io::detail::hdf5::Group::create_group(const std::string& dir) {
  std::string::size_type pos = dir.find_last_of('/');
  if (pos == std::string::npos) { //creates on the current group
    boost::shared_ptr<bob::io::detail::hdf5::Group> g =
      boost::make_shared<bob::io::detail::hdf5::Group>(shared_from_this(), dir);
    m_groups[dir] = g;
    return g;
  }

  //if you get to this point, the search routine needs to be performed on
  //another group, indicated by the path. So, we first cd() there and then do
  //the same as we do here. This will recurse through the directory structure
  //until we find the place defined by the user or raise an exception.
  std::string dest = dir.substr(0, pos);
  if (!dest.size()) dest = "/";
  boost::shared_ptr<bob::io::detail::hdf5::Group> g = cd(dest);
  return g->create_group(dir.substr(pos+1));
}

void bob::io::detail::hdf5::Group::remove_group(const std::string& dir) {
  std::string::size_type pos = dir.find_last_of('/');
  if (pos == std::string::npos) { //copy on the current group
    herr_t status = H5Ldelete(*m_id, dir.c_str(), H5P_DEFAULT);
    if (status < 0) throw bob::io::HDF5StatusError("H5Ldelete", status);
    typedef std::map<std::string, boost::shared_ptr<bob::io::detail::hdf5::Group> > map_type;
    map_type::iterator it = m_groups.find(dir);
    m_groups.erase(it);
    return;
  }

  //if you get to this point, the removal routine needs to be performed on
  //another group, indicated by the path. So, we first cd() there and then do
  //the same as we do here. This will recurse through the directory structure
  //until we find the place defined by the user or raise an exception.
  std::string dest = dir.substr(0, pos);
  if (!dest.size()) dest = "/";
  boost::shared_ptr<bob::io::detail::hdf5::Group> g = cd(dest);
  return g->remove_group(dir.substr(pos+1));
}

/**
 * Opens an "auto-destructible" HDF5 property list
 */
static void delete_h5plist (hid_t* p) {
  if (*p >= 0) {
    herr_t err = H5Pclose(*p);
    if (err < 0) {
      bob::core::error << "H5Pclose() exited with an error (" << err << "). The stack trace follows:" << std::endl;
      bob::core::error << bob::io::format_hdf5_error() << std::endl;
    }
  }
  delete p;
}

static boost::shared_ptr<hid_t> open_plist(hid_t classid) {
  boost::shared_ptr<hid_t> retval(new hid_t(-1), std::ptr_fun(delete_h5plist));
  *retval = H5Pcreate(classid);
  if (*retval < 0) {
    throw bob::io::HDF5StatusError("H5Pcreate", *retval);
  }
  return retval;
}

void bob::io::detail::hdf5::Group::rename_group(const std::string& from, const std::string& to) {
  boost::shared_ptr<hid_t> create_props = open_plist(H5P_LINK_CREATE);
  H5Pset_create_intermediate_group(*create_props, 1);
  herr_t status = H5Lmove(*m_id, from.c_str(), H5L_SAME_LOC, to.c_str(),
      *create_props, H5P_DEFAULT);
  if (status < 0) throw bob::io::HDF5StatusError("H5Lmove", status);
}

void bob::io::detail::hdf5::Group::copy_group(const boost::shared_ptr<Group> other,
    const std::string& dir) {
  std::string::size_type pos = dir.find_last_of('/');
  if (pos == std::string::npos) { //copy on the current group
    const char* use_name = dir.size()?dir.c_str():other->name().c_str();
    herr_t status = H5Ocopy(*other->parent()->location(),
        other->name().c_str(), *m_id, use_name, H5P_DEFAULT, H5P_DEFAULT);
    if (status < 0) throw bob::io::HDF5StatusError("H5Ocopy", status);

    //read new group contents
    boost::shared_ptr<bob::io::detail::hdf5::Group> copied =
      boost::make_shared<bob::io::detail::hdf5::Group>(shared_from_this(), use_name);
    copied->open_recursively();

    //index it
    m_groups[use_name] = copied;

    return;
  }

  //if you get to this point, the copy routine needs to be performed on
  //another group, indicated by the path. So, we first cd() there and then do
  //the same as we do here. This will recurse through the directory structure
  //until we find the place defined by the user or return false.
  std::string dest = dir.substr(0, pos);
  if (!dest.size()) dest = "/";
  boost::shared_ptr<bob::io::detail::hdf5::Group> g = cd(dest);
  return g->copy_group(other, dir.substr(pos+1));
}

bool bob::io::detail::hdf5::Group::has_group(const std::string& dir) const {
  std::string::size_type pos = dir.find_last_of('/');
  if (pos == std::string::npos) { //search on the current group
    if (dir == "." || dir == "..") return true; //special case
    typedef std::map<std::string, boost::shared_ptr<bob::io::detail::hdf5::Group> > map_type;
    map_type::const_iterator it = m_groups.find(dir);
    return (it != m_groups.end());
  }

  //if you get to this point, the search routine needs to be performed on
  //another group, indicated by the path. So, we first cd() there and then do
  //the same as we do here. This will recurse through the directory structure
  //until we find the place defined by the user or return false.
  std::string dest = dir.substr(0, pos);
  if (!dest.size()) dest = "/";
  boost::shared_ptr<bob::io::detail::hdf5::Group> g = cd(dest);
  return g->has_group(dir.substr(pos+1));
}

boost::shared_ptr<bob::io::detail::hdf5::Dataset> bob::io::detail::hdf5::Group::create_dataset
(const std::string& dir, const bob::io::HDF5Type& type, bool list,
 size_t compression) {
  std::string::size_type pos = dir.find_last_of('/');
  if (pos == std::string::npos) { //creates on the current group
    boost::shared_ptr<bob::io::detail::hdf5::Dataset> d =
      boost::make_shared<bob::io::detail::hdf5::Dataset>(shared_from_this(), dir, type,
          list, compression);
    m_datasets[dir] = d;
    return d;
  }

  //if you get to this point, the search routine needs to be performed on
  //another group, indicated by the path. So, we first cd() there and then do
  //the same as we do here. This will recurse through the directory structure
  //until we find the place defined by the user or return false.
  std::string dest = dir.substr(0, pos);
  boost::shared_ptr<bob::io::detail::hdf5::Group> g;
  if (!dest.size()) g = cd("/");
  else {
    //let's make sure the directory exists, or let's create it recursively
    if (!has_group(dest)) g = create_group(dest);
    else g = cd(dest);
  }
  return g->create_dataset(dir.substr(pos+1), type, list, compression);
}

void bob::io::detail::hdf5::Group::remove_dataset(const std::string& dir) {
  std::string::size_type pos = dir.find_last_of('/');
  if (pos == std::string::npos) { //removes on the current group
    herr_t status = H5Ldelete(*m_id, dir.c_str(), H5P_DEFAULT);
    if (status < 0) throw bob::io::HDF5StatusError("H5Ldelete", status);
    typedef std::map<std::string, boost::shared_ptr<bob::io::detail::hdf5::Dataset> > map_type;
    map_type::iterator it = m_datasets.find(dir);
    m_datasets.erase(it);
    return;
  }

  //if you get to this point, the removal routine needs to be performed on
  //another group, indicated by the path. So, we first cd() there and then do
  //the same as we do here. This will recurse through the directory structure
  //until we find the place defined by the user or raise an exception.
  std::string dest = dir.substr(0, pos);
  if (!dest.size()) dest = "/";
  boost::shared_ptr<bob::io::detail::hdf5::Group> g = cd(dest);
  return g->remove_dataset(dir.substr(pos+1));
}

void bob::io::detail::hdf5::Group::rename_dataset(const std::string& from, const std::string& to) {
  boost::shared_ptr<hid_t> create_props = open_plist(H5P_LINK_CREATE);
  H5Pset_create_intermediate_group(*create_props, 1);
  herr_t status = H5Lmove(*m_id, from.c_str(), H5L_SAME_LOC, to.c_str(),
      *create_props, H5P_DEFAULT);
  if (status < 0) throw bob::io::HDF5StatusError("H5Ldelete", status);
}

void bob::io::detail::hdf5::Group::copy_dataset(const boost::shared_ptr<Dataset> other,
    const std::string& dir) {

  std::string::size_type pos = dir.find_last_of('/');
  if (pos == std::string::npos) { //search on the current group
    const char* use_name = dir.size()?dir.c_str():other->name().c_str();
    herr_t status = H5Ocopy(*other->parent()->location(),
        other->name().c_str(), *m_id, use_name, H5P_DEFAULT, H5P_DEFAULT);
    if (status < 0) throw bob::io::HDF5StatusError("H5Ocopy", status);
    //read new group contents
    m_datasets[use_name] = boost::make_shared<bob::io::detail::hdf5::Dataset>(shared_from_this(), use_name);
    return;
  }

  //if you get to this point, the copy routine needs to be performed on
  //another group, indicated by the path. So, we first cd() there and then do
  //the same as we do here. This will recurse through the directory structure
  //until we find the place defined by the user.
  std::string dest = dir.substr(0, pos);
  if (!dest.size()) dest = "/";
  boost::shared_ptr<bob::io::detail::hdf5::Group> g = cd(dest);
  return g->copy_dataset(other, dir.substr(pos+1));
}

bool bob::io::detail::hdf5::Group::has_dataset(const std::string& dir) const {
  std::string::size_type pos = dir.find_last_of('/');
  if (pos == std::string::npos) { //search on the current group
    typedef std::map<std::string, boost::shared_ptr<bob::io::detail::hdf5::Dataset> > map_type;
    map_type::const_iterator it = m_datasets.find(dir);
    return (it != m_datasets.end());
  }

  //if you get to this point, the search routine needs to be performed on
  //another group, indicated by the path. So, we first cd() there and then do
  //the same as we do here. This will recurse through the directory structure
  //until we find the place defined by the user or return false.
  std::string dest = dir.substr(0, pos);
  if (!dest.size()) dest = "/";
  boost::shared_ptr<bob::io::detail::hdf5::Group> g = cd(dest);
  return g->has_dataset(dir.substr(pos+1));
}

void bob::io::detail::hdf5::Group::gettype_attribute(const std::string& name,
    bob::io::HDF5Type& type) const {
  bob::io::detail::hdf5::gettype_attribute(m_id, name, type);
}

bool bob::io::detail::hdf5::Group::has_attribute(const std::string& name) const {
  return bob::io::detail::hdf5::has_attribute(m_id, name);
}

void bob::io::detail::hdf5::Group::delete_attribute (const std::string& name) {
  bob::io::detail::hdf5::delete_attribute(m_id, name);
}

void bob::io::detail::hdf5::Group::read_attribute (const std::string& name,
    const bob::io::HDF5Type& dest_type, void* buffer) const {
  bob::io::detail::hdf5::read_attribute(m_id, name, dest_type, buffer);
}

void bob::io::detail::hdf5::Group::write_attribute (const std::string& name,
    const bob::io::HDF5Type& dest_type, const void* buffer) {
  bob::io::detail::hdf5::write_attribute(m_id, name, dest_type, buffer);
}

void bob::io::detail::hdf5::Group::list_attributes(std::map<std::string, bob::io::HDF5Type>& attributes) const {
  bob::io::detail::hdf5::list_attributes(m_id, attributes);
}

template <> void bob::io::detail::hdf5::Group::set_attribute<std::string>(const std::string& name, const std::string& v) {
  bob::io::HDF5Type dest_type(v);
  write_attribute(name, dest_type, reinterpret_cast<const void*>(v.c_str()));
}

template <> std::string bob::io::detail::hdf5::Group::get_attribute(const std::string& name) const {
  HDF5Type type;
  gettype_attribute(name, type);
  boost::shared_array<char> v(new char[type.shape()[0]+1]);
  v[type.shape()[0]] = 0; ///< null termination
  read_attribute(name, type, reinterpret_cast<void*>(v.get()));
  std::string retval(v.get());
  return retval;
}

bob::io::detail::hdf5::RootGroup::RootGroup(boost::shared_ptr<File> parent):
  bob::io::detail::hdf5::Group(parent),
  m_parent(parent)
{
}

bob::io::detail::hdf5::RootGroup::~RootGroup() {
}

const std::string& bob::io::detail::hdf5::RootGroup::filename() const {
  return m_parent.lock()->filename();
}
