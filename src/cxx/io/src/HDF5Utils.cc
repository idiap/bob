/**
 * @author <a href="mailto:andre.dos.anjos@gmail.com">Andre Anjos</a> 
 * @date Wed  6 Apr 22:43:02 2011 
 *
 * @brief Implements a set of utilities to read HDF5 files. 
 */

#include <boost/make_shared.hpp>
#include "io/HDF5Utils.h"

namespace h5 = Torch::io::detail::hdf5;
namespace io = Torch::io;

/**
 * Opens an "auto-destructible" HDF5 dataset
 */
static void delete_h5dataset (hid_t* p) {
  if (*p >= 0) H5Dclose(*p);
  delete p;
  p=0; 
}

static boost::shared_ptr<hid_t> open_dataset(boost::shared_ptr<h5::File>& file,
    const std::string& path) {
  boost::shared_ptr<hid_t> retval(new hid_t(-1), 
      std::ptr_fun(delete_h5dataset));
  *retval = H5Dopen2(*file->m_id, path.c_str(), H5P_DEFAULT);
  if (*retval < 0) {
    throw io::HDF5StatusError("H5Dopen2", *retval);
  }
  return retval;
}

/**
 * Opens an "auto-destructible" HDF5 datatype
 */
static void delete_h5datatype (hid_t* p) {
  if (*p >= 0) H5Tclose(*p); 
  delete p; 
  p=0; 
}

static boost::shared_ptr<hid_t> open_datatype(boost::shared_ptr<h5::File>& file,
    const std::string& path, const boost::shared_ptr<hid_t>& ds) {
  boost::shared_ptr<hid_t> retval(new hid_t(-1), 
      std::ptr_fun(delete_h5datatype));
  *retval = H5Dget_type(*ds);
  if (*retval < 0) {
    throw io::HDF5StatusError("H5Dget_type", *retval);
  }
  return retval;
}

/**
 * Opens an "auto-destructible" HDF5 property list
 */
static void delete_h5plist (hid_t* p) {
  if (*p >= 0) H5Pclose(*p);
  delete p;
  p=0;
}

static boost::shared_ptr<hid_t> open_plist(hid_t classid) {
  boost::shared_ptr<hid_t> retval(new hid_t(-1), std::ptr_fun(delete_h5plist));
  *retval = H5Pcreate(classid);
  if (*retval < 0) {
    throw io::HDF5StatusError("H5Pcreate", *retval);
  }
  return retval;
}

/**
 * Opens an "auto-destructible" HDF5 dataspace
 */
static void delete_h5dataspace (hid_t* p) {
  if (*p >= 0) H5Sclose(*p);
  delete p;
  p=0; 
}

static boost::shared_ptr<hid_t> open_filespace
(boost::shared_ptr<h5::File>& file, const std::string& path,
 const boost::shared_ptr<hid_t>& ds) {
  boost::shared_ptr<hid_t> retval(new hid_t(-1), std::ptr_fun(delete_h5dataspace));
  *retval = H5Dget_space(*ds);
  if (*retval < 0) throw io::HDF5StatusError("H5Dget_space", *retval);
  return retval;
}

static boost::shared_ptr<hid_t> open_memspace(const io::HDF5Type& t) {
  boost::shared_ptr<hid_t> retval(new hid_t(-1), std::ptr_fun(delete_h5dataspace));
  *retval = H5Screate_simple(t.shape().n(), t.shape().get(), 0);
  if (*retval < 0) throw io::HDF5StatusError("H5Screate_simple", *retval);
  return retval;
}

static void set_memspace(boost::shared_ptr<hid_t> s, const io::HDF5Type& t) {
  herr_t status = H5Sset_extent_simple(*s, t.shape().n(), t.shape().get(), 0);
  if (status < 0) throw io::HDF5StatusError("H5Sset_extent_simple", status);
}

/**
 * Figures out if a dataset is expandible
 */
static bool is_extensible(boost::shared_ptr<hid_t>& space) {

  //has unlimited size on first dimension?
  int rank = H5Sget_simple_extent_ndims(*space);
  if (rank < 0) throw io::HDF5StatusError("H5Sget_simple_extent_ndims", rank);

  io::HDF5Shape maxshape(rank);
  herr_t status = H5Sget_simple_extent_dims(*space, 0, maxshape.get());
  if (status < 0) throw io::HDF5StatusError("H5Sget_simple_extent_dims",status);

  return (maxshape[0] == H5S_UNLIMITED);
}

/**
 * Figures out the extents of a dataset
 */
static io::HDF5Shape get_extents(boost::shared_ptr<hid_t>& space) {
  int rank = H5Sget_simple_extent_ndims(*space);
  if (rank < 0) throw io::HDF5StatusError("H5Sget_simple_extent_ndims", rank);
  //is at least a list of scalars, but could be a list of arrays
  io::HDF5Shape shape(rank);
  herr_t status = H5Sget_simple_extent_dims(*space, shape.get(), 0);
  if (status < 0) throw io::HDF5StatusError("H5Sget_simple_extent_dims",status);
  return shape;
}

/**
 * Creates the extensive list of compatible types for each of possible ways to
 * read/write this dataset.
 */
static void reset_compatibility_list(boost::shared_ptr<hid_t>& space,
    const io::HDF5Type& file_base, std::vector<io::HDF5Descriptor>& descr) {

  if (!file_base.shape()) throw std::length_error("empty HDF5 dataset");

  descr.clear();

  switch (file_base.shape().n()) {

    case 1: ///< file type has 1 dimension
      descr.push_back(io::HDF5Descriptor(file_base.type(),
            file_base.shape()[0], is_extensible(space)));
      break;

    case 2:
    case 3:
    case 4:
    case 5:
      {
        io::HDF5Shape alt = file_base.shape();
        alt <<= 1; ///< contract shape
        descr.push_back(io::HDF5Descriptor(io::HDF5Type(file_base.type(), alt),
              file_base.shape()[0], is_extensible(space)).subselect());
      }
      break;

    default:
      throw io::HDF5UnsupportedDimensionError(file_base.shape().n());
  }

  //can always read the data as a single, non-expandible array
  descr.push_back(io::HDF5Descriptor(file_base, 1, false));
}

h5::Dataset::Dataset(boost::shared_ptr<h5::File>& f,
    const std::string& path) :
  m_parent(f),
  m_path(path),
  m_id(open_dataset(m_parent, m_path)),
  m_dt(open_datatype(m_parent, m_path, m_id)),
  m_filespace(open_filespace(m_parent, m_path, m_id)),
  m_descr(),
  m_memspace()
{
  io::HDF5Type type(m_dt, get_extents(m_filespace));
  reset_compatibility_list(m_filespace, type, m_descr);
  m_memspace = open_memspace(m_descr[0].type);
}

/**
 * Creates and writes an "empty" Dataset in an existing file.
 */
static void create_dataset (boost::shared_ptr<h5::File>& file, 
 const std::string& path, const io::HDF5Type& type, bool list,
 size_t compression) {

  io::HDF5Shape xshape(type.shape());
  
  if (list) { ///< if it is a list, add and extra dimension as dimension 0
    xshape = type.shape();
    xshape >>= 1;
    xshape[0] = 0; ///< no elements for the time being
  }

  io::HDF5Shape maxshape(xshape);
  if (list) maxshape[0] = H5S_UNLIMITED; ///< can expand forever
  
  //creates the data space.
  boost::shared_ptr<hid_t> space(new hid_t(-1), 
      std::ptr_fun(delete_h5dataspace));
  *space = H5Screate_simple(xshape.n(), xshape.get(), maxshape.get());
  if (*space < 0) throw io::HDF5StatusError("H5Screate_simple", *space);

  //creates the property list saying we need the data to be chunked if this is
  //supposed to be a list -- HDF5 only support expandability like this.
  boost::shared_ptr<hid_t> dcpl = open_plist(H5P_DATASET_CREATE);

  //according to the HDF5 manual, chunks have to have the same rank as the
  //array shape.
  io::HDF5Shape chunking(xshape);
  chunking[0] = 1;
  if (list) {
    herr_t status = H5Pset_chunk(*dcpl, chunking.n(), chunking.get());
    if (status < 0) throw io::HDF5StatusError("H5Pset_chunk", status);
  }

  //if the user has decided to compress the dataset, do it with gzip.
  if (compression) {
    if (compression > 9) compression = 9;
    herr_t status = H5Pset_deflate(*dcpl, compression);
    if (status < 0) throw io::HDF5StatusError("H5Pset_deflate", status);
  }

  //our link creation property list for HDF5
  boost::shared_ptr<hid_t> lcpl = open_plist(H5P_LINK_CREATE);
  herr_t status = H5Pset_create_intermediate_group(*lcpl, 1); //1 == true
  if (status < 0)
    throw io::HDF5StatusError("H5Pset_create_intermediate_group", status);

  //please note that we don't define the fill value as in the example, but
  //according to the HDF5 documentation, this value is set to zero by default.

  boost::shared_ptr<hid_t> cls = type.htype();

  //finally create the dataset on the file.
  boost::shared_ptr<hid_t> dataset(new hid_t(-1), 
      std::ptr_fun(delete_h5dataset));
  *dataset = H5Dcreate2(*file->m_id, path.c_str(),
      *cls, *space, *lcpl, *dcpl, H5P_DEFAULT);
  if (*dataset < 0) throw io::HDF5StatusError("H5Dcreate2", *dataset);
}

h5::Dataset::Dataset(boost::shared_ptr<File>& f,
    const std::string& path, const io::HDF5Type& type,
    bool list, size_t compression):
  m_parent(f),
  m_path(path),
  m_id(),
  m_dt(),
  m_filespace(),
  m_descr(),
  m_memspace()
{
  //First, we test to see if we can find the named dataset.
  io::HDF5Error::mute();
  hid_t set_id = H5Dopen2(*m_parent->m_id,m_path.c_str(),H5P_DEFAULT);
  io::HDF5Error::unmute();

  if (set_id < 0) create_dataset(m_parent, m_path, type, list, compression);
  else H5Dclose(set_id); //close it, will re-open it properly

  m_id = open_dataset(m_parent, m_path);
  m_dt = open_datatype(m_parent, m_path, m_id);
  m_filespace = open_filespace(m_parent, m_path, m_id);
  io::HDF5Type file_type(m_dt, get_extents(m_filespace));
  reset_compatibility_list(m_filespace, file_type, m_descr);
  m_memspace = open_memspace(m_descr[0].type);
}

h5::Dataset::~Dataset() { }

size_t h5::Dataset::size () const {
  return m_descr[0].size;
}

size_t h5::Dataset::size (const io::HDF5Type& type) const {
  for (size_t k=0; k<m_descr.size(); ++k) {
    if (m_descr[k].type == type) return m_descr[k].size;
  }
  throw Torch::io::HDF5IncompatibleIO(m_parent->m_path.string(), 
      m_path, m_descr[0].type.str(), type.str());
}

/**
 * Locates a compatible type or returns end().
 */
static std::vector<io::HDF5Descriptor>::iterator 
  find_type_index(std::vector<io::HDF5Descriptor>& descr,
      const io::HDF5Type& user_type) {
  std::vector<io::HDF5Descriptor>::iterator it = descr.begin();
  for (; it != descr.end(); ++it) {
    if (it->type == user_type) break;
  }
  return it;
}

std::vector<io::HDF5Descriptor>::iterator 
h5::Dataset::select (size_t index, const io::HDF5Type& dest) {

  //finds compatibility type
  std::vector<io::HDF5Descriptor>::iterator it = find_type_index(m_descr, dest);

  //if we cannot find a compatible type, we throw
  if (it == m_descr.end()) 
    throw Torch::io::HDF5IncompatibleIO(m_parent->m_path.string(), 
        m_path, m_descr[0].type.str(), dest.str());

  //checks indexing
  if (index >= it->size)
    throw Torch::io::HDF5IndexError(m_parent->m_path.string(), m_path,
        it->size, index);

  set_memspace(m_memspace, it->type);

  it->hyperslab_start[0] = index;

  herr_t status = H5Sselect_hyperslab(*m_filespace, H5S_SELECT_SET, 
      it->hyperslab_start.get(), 0, it->hyperslab_count.get(), 0);
  if (status < 0) throw io::HDF5StatusError("H5Sselect_hyperslab", status);

  return it;
}

void h5::Dataset::read (size_t index, const io::HDF5Type& dest, void* buffer) {

  std::vector<io::HDF5Descriptor>::iterator it = select(index, dest);

  herr_t status = H5Dread(*m_id, *it->type.htype(),
      *m_memspace, *m_filespace, H5P_DEFAULT, buffer);

  if (status < 0) throw io::HDF5StatusError("H5Dread", status);
}

void h5::Dataset::write (size_t index, const io::HDF5Type& dest, 
    const void* buffer) {

  std::vector<io::HDF5Descriptor>::iterator it = select(index, dest);

  herr_t status = H5Dwrite(*m_id, *it->type.htype(),
      *m_memspace, *m_filespace, H5P_DEFAULT, buffer);

  if (status < 0) throw io::HDF5StatusError("H5Dwrite", status);
}
    
void h5::Dataset::extend (const Torch::io::HDF5Type& dest, const void* buffer) {
  
  //finds compatibility type
  std::vector<io::HDF5Descriptor>::iterator it = find_type_index(m_descr, dest);

  //if we cannot find a compatible type, we throw
  if (it == m_descr.end()) 
    throw Torch::io::HDF5IncompatibleIO(m_parent->m_path.string(), 
        m_path, m_descr[0].type.str(), dest.str());

  if (!it->expandable)
    throw io::HDF5NotExpandible(m_parent->m_path.string(), m_path);

  //if it is expandible, try expansion
  io::HDF5Shape tmp(it->type.shape());
  tmp >>= 1;
  tmp[0] = it->size + 1;
  herr_t status = H5Dset_extent(*m_id, tmp.get());
  if (status < 0) throw io::HDF5StatusError("H5Dset_extent", status);
  
  //if expansion succeeded, update all compatible types
  for (size_t k=0; k<m_descr.size(); ++k) {
    if (m_descr[k].expandable) { //updated only the length
      m_descr[k].size += 1;
    }
    else { //not expandable, update the shape/count for a straight read/write
      m_descr[k].type.shape()[0] += 1;
      m_descr[k].hyperslab_count[0] += 1;
    }
  }
      
  m_filespace = open_filespace(m_parent, m_path, m_id); //update filespace

  write(tmp[0]-1, dest, buffer);
}

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
    *retval = H5Fcreate(path.string().c_str(), H5F_ACC_TRUNC, *fcpl, H5P_DEFAULT);
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

void h5::File::unlink (const std::string& path) {
  herr_t status = H5Ldelete(*m_id, path.c_str(), H5P_DEFAULT);
  if (status < 0) throw io::HDF5StatusError("H5Ldelete", status);
  
  //TODO: Recursively erase empty groups.
}

void h5::File::rename (const std::string& from,
    const std::string& to) {
  //our link creation property list for HDF5
  boost::shared_ptr<hid_t> lcpl = open_plist(H5P_LINK_CREATE);
  herr_t status = H5Pset_create_intermediate_group(*lcpl, 1); //1 == true
  if (status < 0) 
    throw io::HDF5StatusError("H5Pset_create_intermediate_group", status);

  status = H5Lmove(*m_id, from.c_str(), *m_id, to.c_str(), *lcpl, H5P_DEFAULT);

  if (status < 0) throw io::HDF5StatusError("H5Lmove", status);
  
  //TODO: Recursively erase empty groups.
}

size_t h5::File::userblock_size() const {
  hsize_t retval;
  herr_t err = H5Pget_userblock(*m_fcpl, &retval);
  if (err < 0) throw io::HDF5StatusError("H5Pget_create_plist", err);
  return retval;
}

/**
 * Callback function. Given a group/dataset in the opened HDF5 file, creates 
 * a corresponding entry in the index dictionary, if it is an HDF5 hard link
 * to a dataset.
 */
static herr_t fill_index_callback(hid_t group, const char *name, 
    const H5L_info_t *info, void *cpair) {
  // Gets the H5File pointer
  std::pair<boost::shared_ptr<h5::File>, 
    std::map<std::string, boost::shared_ptr<h5::Dataset> >* >*
      cookie = (std::pair<boost::shared_ptr<h5::File>, 
          std::map<std::string, boost::shared_ptr<h5::Dataset> >* >*)cpair;
  boost::shared_ptr<h5::File>& file = cookie->first;
  std::map<std::string, boost::shared_ptr<h5::Dataset> >& dict=*cookie->second;

  // If we are not looking at a hard link to the data, just ignore
  if (info->type != H5L_TYPE_HARD) return 0;

  // Get information about the HDF5 object
  H5O_info_t obj_info;
  herr_t status = H5Oget_info_by_name(group, name, &obj_info, H5P_DEFAULT);
  if (status < 0) throw io::HDF5StatusError("H5Oget_info_by_name", status);

  // If the object we are currently reading is not a Dataset, just ignore
  if (obj_info.type != H5O_TYPE_DATASET) return 0;

  std::string complete("/");
  complete += name;
  dict[complete].reset(new h5::Dataset(file, complete));
  return 0;
}

void h5::index(boost::shared_ptr<h5::File>& file,
    std::map<std::string, boost::shared_ptr<h5::Dataset> >& index) {
  //iterate over all leafs in HDF5/C-style using callbacks.
  std::pair<boost::shared_ptr<h5::File>, 
    std::map<std::string, boost::shared_ptr<h5::Dataset> >* >
      cookie = std::make_pair(file, &index);
  herr_t status = H5Lvisit(*file->m_id, H5_INDEX_NAME, 
      H5_ITER_NATIVE, fill_index_callback, &cookie);
  if (status < 0) throw io::HDF5StatusError("H5Lvisit_by_name", status);
}
