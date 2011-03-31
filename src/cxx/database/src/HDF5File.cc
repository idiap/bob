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
 * Given an opened HDF5 file, fills the index dictionary with all (leaf) paths
 * to HDF5 Datasets. This method will do a recursive walk through the file
 * hierarchy and will just get the leafs out of it.
 */
static void fill_index(H5::H5File file, std::map<boost::filesystem::path, db::HDF5File::typeinfo>& index) {
  //TODO
  //std::cout << endl << "Iterating over elements in the file" << endl;
  //herr_t idx = H5Literate(file->getId(), H5_INDEX_NAME, H5_ITER_INC, NULL, file_info, NULL);
  //std::cout << endl
  return;
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
