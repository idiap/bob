/**
 * @author <a href="mailto:andre.dos.anjos@gmail.com">Andre Anjos</a> 
 * @date Thu 31 Mar 07:42:23 2011 
 *
 * @brief Implementation of the read/write functionality for HDF5 files 
 */

#include "database/HDF5File.h"

namespace db = Torch::database;

static unsigned int getH5Access (db::HDF5File::mode_t v) {
  switch(v)
  {
    case 0: return H5F_ACC_RDONLY;
    case 1: return H5F_ACC_RDWR;
    case 2: return H5F_ACC_TRUNC;
    case 4: return H5F_ACC_EXCL;
    default:
      throw db::HDF5InvalidFileAccessModeError(v);
  }
}

db::HDF5File::HDF5File(const std::string& filename, mode_t mode):
  m_file(new db::detail::hdf5::File(filename, getH5Access(mode))),
  m_index()
{
  //makes sure we will shut-up the HDF5 automatic logging before we start
  static boost::shared_ptr<db::HDF5Error> init = db::HDF5Error::instance();
  db::detail::hdf5::index(m_file, m_index);   
}

db::HDF5File::~HDF5File() {
}

bool db::HDF5File::contains (const std::string& path) const {
  return m_index.find(path) != m_index.end();
}

const db::HDF5Type& db::HDF5File::describe (const std::string& path) const {
  if (!contains(path)) throw db::HDF5InvalidPath(m_file->m_path.string(), path);
  return m_index.find(path)->second->m_type;
}

void db::HDF5File::unlink (const std::string& path) {
  if (!contains(path)) throw db::HDF5InvalidPath(m_file->m_path.string(), path);
  m_file->unlink(path);
  m_index.erase(path);
}

void db::HDF5File::rename (const std::string& from, const std::string& to) {
  if (!contains(from)) throw db::HDF5InvalidPath(m_file->m_path.string(), from);
  m_file->rename(from, to);
  m_index[to] = m_index[from];
  m_index.erase(from);
}

size_t db::HDF5File::size (const std::string& path) const {
  if (!contains(path)) throw db::HDF5InvalidPath(m_file->m_path.string(), path);
  return m_index.find(path)->second->size();
}
      
void db::HDF5File::copy (HDF5File& other) {
  //TODO
}
