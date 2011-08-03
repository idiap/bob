/**
 * @author <a href="mailto:andre.dos.anjos@gmail.com">Andre Anjos</a> 
 * @date Wed  6 Apr 22:43:02 2011 
 *
 * @brief Implements a set of utilities to read HDF5 files. 
 */

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
 * Opens an "auto-destructible" HDF5 file dataspace
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

/**
 * Figures out if a dataset is chunked and therefore expandible
 */
static bool is_chunked(boost::shared_ptr<hid_t>& dataset) {
  hid_t plist = H5Dget_create_plist(*dataset);
  if (plist < 0) throw io::HDF5StatusError("H5Pget_create_plist", plist);
  bool retval (H5D_CHUNKED == H5Pget_layout(plist));
  H5Pclose(plist);
  return retval;
}

/**
 * Figures out the extents of a dataset
 *
 * TODO: Check extendibility of the first dimension, check on HDF5Shape that
 * the type is extendible or not. If type is extendible, may be read as
 * D-dimensional object or as a list of objects of dimension D-1.
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

h5::Dataset::Dataset(boost::shared_ptr<h5::File>& f,
    const std::string& path) :
  m_parent(f),
  m_path(path),
  m_id(open_dataset(m_parent, m_path)),
  m_dt(open_datatype(m_parent, m_path, m_id)),
  m_filespace(open_filespace(m_parent, m_path, m_id)),
  m_memspace(),
  m_extent(get_extents(m_filespace)),
  m_select_offset(m_extent.n()),
  m_select_count(m_extent),
  m_chunked(is_chunked(m_id)),
  m_type(m_dt, m_extent)
{
  m_select_count[0] = 1;
  m_memspace = open_memspace(m_type);
}

/**
 * Creates and writes a "empty" Dataset in an existing file.
 */
static void create_dataset (boost::shared_ptr<h5::File>& file, 
 const std::string& path, const io::HDF5Type& type,
 size_t compression) {

  io::HDF5Shape xshape(type.shape().n()+1);
  for (size_t i=0; i<type.shape().n(); ++i) xshape[i+1] = type.shape()[i];
  xshape[0] = 0; ///< no elements for the time being

  io::HDF5Shape maxshape(xshape);
  maxshape[0] = H5S_UNLIMITED; ///< can expand forever
  
  //creates the data space. first dimension => length of the set
  boost::shared_ptr<hid_t> space(new hid_t(-1), std::ptr_fun(delete_h5dataspace));
  *space = H5Screate_simple(type.shape().n()+1, xshape.get(), 
      maxshape.get());
  if (*space < 0) throw io::HDF5StatusError("H5Screate_simple", *space);

  //creates the property list saying we need the data to be chunked. hdf5 only
  //support expandability like this.
  boost::shared_ptr<hid_t> dcpl = open_plist(H5P_DATASET_CREATE);

  //according to the HDF5 manual, chunks have to have the same rank as the
  //array shape.
  io::HDF5Shape chunking(xshape);
  chunking[0] = 1;
  herr_t status = H5Pset_chunk(*dcpl, chunking.n(), chunking.get());
  if (status < 0) throw io::HDF5StatusError("H5Pset_chunk", status);

  //if the user has decided to compress the dataset, do it with gzip.
  if (compression) {
    if (compression > 9) compression = 9;
    status = H5Pset_deflate(*dcpl, compression);
    if (status < 0) throw io::HDF5StatusError("H5Pset_deflate", status);
  }

  //our link creation property list for HDF5
  boost::shared_ptr<hid_t> lcpl = open_plist(H5P_LINK_CREATE);
  status = H5Pset_create_intermediate_group(*lcpl, 1); //1 == true
  if (status < 0) 
    throw io::HDF5StatusError("H5Pset_create_intermediate_group", status);

  //please note that we don't define the fill value as in the example, but
  //according to the HDF5 documentation, this value is set to zero by default.

  boost::shared_ptr<hid_t> cls = type.htype();

  //finally create the dataset on the file.
  boost::shared_ptr<hid_t> dataset(new hid_t(-1), std::ptr_fun(delete_h5dataset));
  *dataset = H5Dcreate2(*file->m_id, path.c_str(),
      *cls, *space, *lcpl, *dcpl, H5P_DEFAULT);
  if (*dataset < 0) throw io::HDF5StatusError("H5Dcreate2", *dataset);
}

h5::Dataset::Dataset(boost::shared_ptr<File>& f,
    const std::string& path, const io::HDF5Type& type,
    size_t compression):
  m_parent(f),
  m_path(path),
  m_id(),
  m_dt(),
  m_filespace(),
  m_memspace(),
  m_extent(),
  m_select_offset(),
  m_select_count(),
  m_chunked(false),
  m_type(type)
{
  //First, we test to see if we can find the named dataset.
  io::HDF5Error::mute();
  hid_t set_id = H5Dopen2(*m_parent->m_id,m_path.c_str(),H5P_DEFAULT);
  io::HDF5Error::unmute();

  if (set_id < 0) create_dataset(m_parent, m_path, m_type, compression);
  else H5Dclose(set_id); //close it, will re-open it properly

  m_id = open_dataset(m_parent, m_path);
  m_dt = open_datatype(m_parent, m_path, m_id);
  m_filespace = open_filespace(m_parent, m_path, m_id);
  m_extent = get_extents(m_filespace);
  m_select_offset = HDF5Shape(m_extent.n());
  m_select_count = m_extent;
  m_select_count[0] = 1;
  m_type = io::HDF5Type(m_dt, m_extent);
  m_memspace = open_memspace(m_type);
  m_chunked = is_chunked(m_id);
}

h5::Dataset::~Dataset() { }

void h5::Dataset::select (size_t index) {
  m_select_offset[0] = index;
  herr_t status = H5Sselect_hyperslab(*m_filespace, H5S_SELECT_SET, 
      m_select_offset.get(), 0, m_select_count.get(), 0);
  if (status < 0) throw io::HDF5StatusError("H5Sselect_hyperslab", status);
}

void h5::Dataset::read (void* buffer) {
  boost::shared_ptr<hid_t> htype = m_type.htype();
  herr_t status = H5Dread(*m_id, *htype, *m_memspace, *m_filespace,
      H5P_DEFAULT, buffer);
  if (status < 0) throw io::HDF5StatusError("H5Dread", status);
}

void h5::Dataset::extend () {
  m_extent[0] += 1;
  herr_t status = H5Dset_extent(*m_id, m_extent.get());
  if (status < 0) {
    m_extent[0] -= 1;
    throw io::HDF5StatusError("H5Dset_extent", status);
  }
  m_filespace = open_filespace(m_parent, m_path, m_id); //update filespace
}

void h5::Dataset::write (const void* buffer) {
  boost::shared_ptr<hid_t> htype = m_type.htype();
  herr_t status = H5Dwrite(*m_id, *htype, *m_memspace, *m_filespace,
      H5P_DEFAULT, buffer);
  if (status < 0) throw io::HDF5StatusError("H5Dwrite", status);
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
