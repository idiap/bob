/**
 * @file cxx/io/io/HDF5Group.h
 * @author Andre Anjos <andre.anjos@idiap.ch>
 * @date Wed 29 Feb 17:24:10 2012
 *
 * @brief Describes HDF5 groups.
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

#ifndef BOB_IO_HDF5GROUP_H
#define BOB_IO_HDF5GROUP_H

#include <boost/shared_ptr.hpp>
#include <boost/enable_shared_from_this.hpp>
#include <hdf5.h>
#include "io/HDF5Types.h"
#include "io/HDF5Dataset.h"

namespace bob { namespace io { namespace detail { namespace hdf5 {

  class File;

  /**
   * A group represents a path inside the HDF5 file. It can contain Datasets or
   * other Groups.
   */
  class Group: public boost::enable_shared_from_this<Group> {

    public: //better to protect?

      /**
       * Creates a new group in a given parent.
       */
      Group(boost::shared_ptr<Group> parent, const std::string& name);

      /**
       * Binds to an existing group in a parent, reads all the group contents
       * recursively. Note that the last parameter is there only to
       * differentiate from the above constructor. It is ignored.
       */
      Group(boost::shared_ptr<Group> parent,  const std::string& name,
          bool open);

      /**
       * Constructor used by the root group, just open the root group
       */
      Group(boost::shared_ptr<File> parent);

      /**
       * Recursively open sub-groups, attributes and datasets. This cannot be
       * done at the constructor because of a enable_shared_from_this<>
       * restriction that results in a bad weak pointer exception being raised.
       */
      void open_recursively();

    public: //api

      /**
       * D'tor - presently, does nothing
       */
      virtual ~Group();

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
       * Deletes all children nodes and properties in this group.
       *
       * Note that removing data already written in a file will only be
       * effective in terms of space saving when you actually re-write that
       * file. This instruction just unlinks all data from this group and makes
       * them inaccessible to any further read operation.
       */
      virtual void reset();

      /**
       * Accesses the current location id of this group
       */
      const boost::shared_ptr<hid_t> location() const {
        return m_id;
      }

      boost::shared_ptr<hid_t> location() {
        return m_id;
      }

      /**
       * move up-down on the group hierarchy
       */
      virtual boost::shared_ptr<Group> cd(const std::string& path);
      virtual const boost::shared_ptr<Group> cd(const std::string& path) const;

      /**
       * Get a mapping of all child groups
       */
      virtual const std::map<std::string, boost::shared_ptr<Group> >& groups()
        const {
        return m_groups;
      }

      /**
       * Create a new subgroup with a given name.
       */
      virtual boost::shared_ptr<Group> create_group(const std::string& name);

      /**
       * Deletes an existing subgroup with a given name. If a relative name is
       * given, it is interpreted w.r.t. to this group.
       *
       * Note that removing data already written in a file will only be
       * effective in terms of space saving when you actually re-write that
       * file. This instruction just unlinks all data from this group and makes
       * them inaccessible to any further read operation.
       */
      virtual void remove_group(const std::string& path);

      /**
       * Rename an existing group under me.
       */
      virtual void rename_group(const std::string& from, const std::string& to);

      /**
       * Copies all data from an existing group into myself, creating a new
       * subgroup, by default, with the same name as the other group. If a
       * relative name is given, it is interpreted w.r.t. to this group.
       *
       * If an empty string is given as "dir", copies the other group name.
       */
      virtual void copy_group(const boost::shared_ptr<Group> other, const
          std::string& path="");

      /**
       * Says if a group with a certain path exists in this group.
       */
      virtual bool has_group(const std::string& path) const;

      /**
       * Get all datasets attached to this group
       */
      virtual const std::map<std::string, boost::shared_ptr<Dataset> >&
        datasets() const {
          return m_datasets;
        }

      /**
       * Creates a new HDF5 dataset from scratch and inserts it in this group.
       * If the Dataset already exists on file and the types are compatible, we
       * attach to that type, otherwise, we raise an exception.
       *
       * You can set if you would like to have the dataset created as a list
       * and the compression level.
       *
       * The effect of setting "list" to false is that the created dataset:
       *
       * a) Will not be expandible (chunked) b) Will contain the exact number
       * of dimensions of the input type.
       *
       * When you set "list" to true (the default), datasets are created with
       * chunking automatically enabled (the chunk size is set to the size of
       * the given variable) and an extra dimension is inserted to accomodate
       * list operations.
       */
      virtual boost::shared_ptr<Dataset> create_dataset
        (const std::string& path, const bob::io::HDF5Type& type, bool list=true,
         size_t compression=0);

      /**
       * Deletes a dataset in this group
       *
       * Note that removing data already written in a file will only be
       * effective in terms of space saving when you actually re-write that
       * file. This instruction just unlinks all data from this group and makes
       * them inaccessible to any further read operation.
       */
      virtual void remove_dataset(const std::string& path);

      /**
       * Rename an existing dataset under me.
       */
      virtual void rename_dataset(const std::string& from,
          const std::string& to);

      /**
       * Copies the contents of the given dataset into this. By default, use
       * the same name.
       */
      virtual void copy_dataset(const boost::shared_ptr<Dataset> other,
          const std::string& path="");

      /**
       * Says if a dataset with a certain name exists in the current file.
       */
      virtual bool has_dataset(const std::string& path) const;

      /**
       * Accesses a certain dataset from this group
       */
      boost::shared_ptr<Dataset> operator[] (const std::string& path);
      const boost::shared_ptr<Dataset> operator[] (const std::string& path) const;

      /**
       * Accesses all existing paths in one shot. Input has to be a std
       * container with T = std::string and accepting push_back()
       */
      template <typename T> void dataset_paths (T& container) const {
        for (std::map<std::string, boost::shared_ptr<io::detail::hdf5::Dataset> >::const_iterator it=m_datasets.begin(); it != m_datasets.end(); ++it) container.push_back(it->second->path());
        for (std::map<std::string, boost::shared_ptr<io::detail::hdf5::Group> >::const_iterator it=m_groups.begin(); it != m_groups.end(); ++it) it->second->dataset_paths(container);
      }

      /**
       * Callback function for group iteration. Two cases are blessed here:
       *
       * 1. Object is another group. In this case just instantiate the group and
       *    recursively iterate from there
       * 2. Object is a dataset. Instantiate it.
       *
       * Only hard-links are considered. At the time being, no soft links.
       */
      herr_t iterate_callback(hid_t group, const char *name,
          const H5L_info_t *info);

    public: //attribute hack

      /**
       * Sets a scalar attribute on the current group. Setting an existing
       * attribute overwrites its value.
       *
       * @note Only simple scalars are supported for the time being
       */
      template <typename T> void set_attribute(const std::string& name, 
          const T& v) {
        bob::io::HDF5Type dest_type(v);
        write_attribute(name, dest_type, reinterpret_cast<const void*>(&v));
      }

      /**
       * Reads an attribute from the current group. Raises an error if such
       * attribute does not exist on the group. To check for existence, use
       * has_attribute().
       */
      template <typename T> T get_attribute(const std::string& name) const {
        T v;
        bob::io::HDF5Type dest_type(v);
        read_attribute(name, dest_type, reinterpret_cast<void*>(&v));
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

    private: //attribute setting/getting private methods

      /**
       * reads the attribute value, place it in "buffer"
       */
      void read_attribute (const std::string& name,
          const bob::io::HDF5Type& dest, void* buffer) const;

      /**
       * writes an attribute value from "buffer"
       */
      void write_attribute (const std::string& name,
          const bob::io::HDF5Type& dest, const void* buffer);

    private: //not implemented

      /**
       * Copies the contents of an existing group -- not implemented
       */
      Group(const Group& other);

      /**
       * Assigns the contents of an existing group to myself -- not
       * implemented
       */
      Group& operator= (const Group& other);

    private: //representation

      std::string m_name; ///< my name
      boost::shared_ptr<hid_t> m_id; ///< the HDF5 Group this object points to
      boost::weak_ptr<Group> m_parent;
      std::map<std::string, boost::shared_ptr<Group> > m_groups;
      std::map<std::string, boost::shared_ptr<Dataset> > m_datasets;
      //std::map<std::string, boost::shared_ptr<Attribute> > m_attributes;

  };

  /**
   * The RootGroup is a special case of the Group object that is directly
   * attached to the File (no parents).
   */
  class RootGroup: public Group {

    public: //api

      /**
       * Binds to the root group of a file.
       */
      RootGroup(boost::shared_ptr<File> parent);

      /**
       * D'tor - presently, does nothing
       */
      virtual ~RootGroup();

      /**
       * Get parent group
       */
      virtual const boost::shared_ptr<Group> parent() const {
        return boost::shared_ptr<Group>();
      }

      /**
       * Get parent group
       */
      virtual boost::shared_ptr<Group> parent() {
        return boost::shared_ptr<Group>();
      }

      /**
       * Filename where I'm sitting
       */
      virtual const std::string& filename() const;

      /**
       * Full path to myself. Constructed each time it is called.
       */
      virtual std::string path() const {
        return "";
      }

      /**
       * Access file
       */
      virtual const boost::shared_ptr<File> file() const {
        return m_parent.lock();
      }

      virtual boost::shared_ptr<File> file() {
        return m_parent.lock();
      }

    private: //representation

      boost::weak_ptr<File> m_parent; ///< the file I belong to

  };

}}}}

#endif /* BOB_IO_HDF5GROUP_H */
