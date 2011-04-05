/**
 * @author <a href="mailto:andre.dos.anjos@gmail.com">Andre Anjos</a> 
 * @date Thu 31 Mar 07:42:23 2011 
 *
 * @brief Implementation of the read/write functionality for HDF5 files 
 */

#include "database/HDF5File.h"

namespace db = Torch::database;

static unsigned int getH5Access( db::HDF5File::mode_t v) {
  switch(v)
  {
    case 0: return H5F_ACC_RDONLY;
    case 1: return H5F_ACC_RDWR;
    case 2: return H5F_ACC_TRUNC;
    case 4: return H5F_ACC_EXCL;
    default: //TODO: Combination of flags?
      throw db::HDF5InvalidFileAccessModeError(v);
  }
}


/**
 * Callback function. Given a group/dataset in the opened HDF5 file, creates 
 * a corresponding entry in the index dictionary, if it is an HDF5 hard link
 * to a dataset.
 */
static herr_t fill_index_callback( hid_t g_id, const char *name, const H5L_info_t *info, void *op_data) {
  std::cout << name << std::endl;

  // Get the H5File pointer
  H5::H5File *myfile = (H5::H5File*)op_data;

  // Check that it is a hard link
  if(info->type == H5L_TYPE_HARD) {
    H5O_info_t  oinfo;

    // Get information about the HDF5 object
    if(H5Oget_info_by_name(g_id, name, &oinfo, H5P_DEFAULT) < 0)
      throw Torch::database::Exception(); // TODO: add a specialized exception

    // If it is a dataset, get the datatype and add it to the index.
    if(oinfo.type == H5O_TYPE_DATASET)
    {
      // Get the dataset identifier
      hid_t dataset_id = H5Dopen(myfile->getId(), name, H5P_DEFAULT);
      // Get the datatype identifier
      hid_t dtype_id = H5Dget_type(dataset_id);
      // Get the dataspace identifier
      hid_t dspace_id = H5Dget_space(dataset_id);

      // TODO: Generate the full typeinfo for this dataset
      //db::HDF5File::typeinfo tinfo;

      // Parse the type from the HDF5 dataset/datatype
      // Get the class
      switch (H5Tget_class(dtype_id)) {
        case H5T_INTEGER:
          break;
        case H5T_FLOAT:
          break;
        case H5T_STRING:
          break;
        // TODO: complex? boolean? HDF5 dataset-level Array?
        default: 
          throw Torch::database::Exception(); // TODO: add a specialized exception
      }
      // Get signed/unsigned
      switch(H5Tget_sign(dtype_id)) {
        case H5T_SGN_NONE: // unsigned
          break;
        case H5T_SGN_2: // signed
          break;
        default:
          break;
      }
      // Get the size
      size_t sz = H5Tget_size(dtype_id);
      // TODO: Determine the type based on previous information


      // Parse the shape from the HDF5 dataset/dataspace
      int rank = H5Sget_simple_extent_ndims(dspace_id);
      hsize_t *current_dims= (hsize_t*)malloc(rank*sizeof(hsize_t));
      hsize_t *max_dims=(hsize_t*)malloc(rank*sizeof(hsize_t));
      if( rank != H5Sget_simple_extent_dims(dspace_id, current_dims, max_dims) )
        throw Torch::database::Exception(); // TODO: add a specialized exception

      // Parse the number of elements
      int n_elem = H5Sget_simple_extent_npoints(dspace_id);

      // Free memory allocation
      free(current_dims);
      free(max_dims);

      // Close the dataset
      if( H5Dclose(dataset_id )  < 0)
        throw Torch::database::Exception(); // TODO: add a specialized exception
    }
  }

  return 0;
}

/**
 * Given an opened HDF5 file, fills the index dictionary with all (leaf) paths
 * to HDF5 Datasets. This method will do a recursive walk through the file
 * hierarchy and will just get the leafs out of it. This is done by using the
 * H5Lvisit_by_name() function of the C API.
 */
static void fill_index(H5::H5File file, std::map<boost::filesystem::path, db::HDF5File::typeinfo>& index) {
  herr_t v_returned= H5Lvisit_by_name(file.getLocId(), "/", H5_INDEX_NAME, H5_ITER_NATIVE, fill_index_callback, &file, H5P_DEFAULT);
  if( v_returned < 0)
    throw Torch::database::Exception(); // TODO: Add a new dedicated exception
}

/**
 * Turns off auto-printing for HDF5 exceptions
 */
static bool hdf5_configure() {
  H5::Exception::dontPrint();
  return true;
}

db::HDF5File::HDF5File(const boost::filesystem::path& filename, mode_t mode):
  m_path(filename),
  m_file(),
  m_index()
{
  static bool configured = hdf5_configure(); //do once HDF5 configuration items

  //this may raise H5::Exception's with error stacks if any problem is found.
  try {
    m_file.openFile(m_path.string().c_str(), getH5Access(mode));
    if (mode == db::HDF5File::in || mode == db::HDF5File::inout) 
      fill_index(m_file, m_index);
  }
  catch (H5::Exception& e) {
    //TODO: Transform this exception in a standard Torch exception, re-raise
  }
}

db::HDF5File::~HDF5File() {
}


bool db::HDF5File::contains(const boost::filesystem::path& path) {
  return m_index.find(path) != m_index.end();
}

const db::HDF5File::typeinfo& db::HDF5File::describe(const boost::filesystem::path& path) {
  if (contains(path)) return m_index.find(path)->second;
  //TODO: Raise NotFound
  throw db::HDF5ObjectNotFoundError( path.string(), m_file.getFileName() );
}

void db::HDF5File::unlink(const boost::filesystem::path& path) {
  //unlink HDF5 file element
  //remove m_index entry
}

void db::HDF5File::copy(const db::HDF5File::HDF5File& other) {
  for (std::map<boost::filesystem::path, db::HDF5File::typeinfo>::const_iterator it = other.m_index.begin(); it != other.m_index.end(); ++it) {
    /*
    if (!it->second.rank) {
      //TODO: do one of those for every supported scalar type T
      addScalar(it->first, other.getScalar<T>(it->first));
    }
    else {
      //TODO: do one of those for every supported array type T
      addArray(it->first, other.getArray<T>(it->first));
    }
    */
  }
}
