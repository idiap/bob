/**
 * @author <a href="mailto:andre.dos.anjos@gmail.com">Andre Anjos</a> 
 * @date Thu 31 Mar 07:42:23 2011 
 *
 * @brief Implementation of the read/write functionality for HDF5 files 
 */

#include "database/HDF5File.h"

namespace db = Torch::database;
namespace fs = boost::filesystem;

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
  m_index(),
  m_cwd("/") ///< we start by looking at the root directory
{
  //makes sure we will shut-up the HDF5 automatic logging before we start
  db::detail::hdf5::index(m_file, m_index);
}

db::HDF5File::~HDF5File() {
}

void db::HDF5File::cd(const std::string& path) {
  m_cwd = resolve(path);
}

const std::string& db::HDF5File::cwd() const {
  return m_cwd.string();
}

static fs::path trim_one(const fs::path& p) {
  if (p == p.root_path()) return p;
  fs::path retval;
  for (fs::path::iterator it = p.begin(); it!=p.end(); ++it) {
    fs::path::iterator next = it;
    ++next; //< for the lack of better support in boost::filesystem V2
    if (next == p.end()) break; //< == skip the last bit
    retval /= *it;
  }
  return retval;
}

std::string db::HDF5File::resolve(const std::string& path) const {
  //the path to be solved is what the user inputs, unless (s)he inputs a
  //relative path, in which case we complete from him/her.
  fs::path completed(path);
  if (! completed.is_complete()) completed = fs::complete(completed, m_cwd);

  //now we prune the path to make sure we don't have relative bits inside, like
  //'..' or '.'
  fs::path retval;
  for (fs::path::iterator it = completed.begin(); it != completed.end(); ++it) {
    if (*it == "..") {
      retval = trim_one(retval);
      continue;
    }
    if (*it == ".") { //ignore '.'
      continue;
    }
    retval /= *it;
  }
  return retval.string();
}

bool db::HDF5File::contains (const std::string& path) const {
  return m_index.find(resolve(path)) != m_index.end();
}

const db::HDF5Type& db::HDF5File::describe (const std::string& path) const {
  std::string absolute = resolve(path);
  if (!contains(path)) throw db::HDF5InvalidPath(m_file->m_path.string(), absolute);
  return m_index.find(absolute)->second->m_type;
}

void db::HDF5File::unlink (const std::string& path) {
  std::string absolute = resolve(path);
  if (!contains(path)) throw db::HDF5InvalidPath(m_file->m_path.string(), absolute);
  m_file->unlink(absolute);
  m_index.erase(absolute);
}

void db::HDF5File::rename (const std::string& from, const std::string& to) {
  std::string absfrom = resolve(from);
  std::string absto = resolve(to);
  if (!contains(absfrom)) 
    throw db::HDF5InvalidPath(m_file->m_path.string(), absfrom);
  m_file->rename(absfrom, absto);
  m_index[absto] = m_index[absfrom];
  m_index.erase(absfrom);
}

size_t db::HDF5File::size (const std::string& path) const {
  std::string absolute = resolve(path);
  if (!contains(absolute)) throw db::HDF5InvalidPath(m_file->m_path.string(), absolute);
  return m_index.find(absolute)->second->size();
}
      
void db::HDF5File::copy (HDF5File& other) {
  //TODO
}
