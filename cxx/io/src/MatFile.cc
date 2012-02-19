/**
 * @file cxx/io/src/MatFile.cc
 * @date Wed Oct 26 17:11:16 2011 +0200
 * @author Andre Anjos <andre.anjos@idiap.ch>
 *
 * @brief Implements the matlab (.mat) array codec using matio
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
#include <boost/filesystem.hpp>
#include <boost/format.hpp>
#include <algorithm>

#include "io/MatUtils.h"
#include "io/CodecRegistry.h"
#include "io/Exception.h"

namespace fs = boost::filesystem;
namespace io = bob::io;
namespace ca = bob::core::array;

/**
 * TODO:
 * 1. Current known limitation: does not support full read-out of all data if
 * an array_read() is issued. What we do, presently, is just to read the first
 * variable.
 */
class MatFile: public io::File {

  public: //api

    MatFile(const std::string& path, char mode):
      m_filename(path),
      m_mode( (mode=='r')? MAT_ACC_RDONLY : MAT_ACC_RDWR ),
      m_map(new std::map<size_t, std::pair<std::string, ca::typeinfo> >()),
      m_size(0) {
        if (mode == 'r' || mode == 'a') try_reload_map();
        if (mode == 'w' && fs::exists(path)) fs::remove(path);
      }

    virtual ~MatFile() { }

    void try_reload_map () {
      if (fs::exists(m_filename)) {
        m_map = io::detail::list_variables(m_filename);
        m_type = m_map->begin()->second.second;
        m_size = m_map->size();
        m_id.reserve(m_size);
        for (map_type::iterator
            it = m_map->begin(); it != m_map->end(); ++it) {
          m_id.push_back(it->first);
        }
        std::sort(m_id.begin(), m_id.end()); //get the right order...

        //double checks some parameters
        if (m_type.nd == 0 || m_type.nd > 4) 
          throw io::DimensionError(m_type.nd, BOB_MAX_DIM);
        if (m_type.dtype == bob::core::array::t_unknown) 
          throw io::UnsupportedTypeError(m_type.dtype);
      }
    }

    virtual const std::string& filename() const {
      return m_filename;
    }

    virtual const ca::typeinfo& array_type () const {
      return m_type;
    }

    virtual const ca::typeinfo& arrayset_type () const {
      return m_type;
    }

    virtual size_t arrayset_size() const {
      return m_size;
    }

    virtual const std::string& name() const {
      return s_codecname;
    }

    virtual void array_read(ca::interface& buffer) {
      
      //do we need to reload the file?
      if (!m_type.is_valid()) try_reload_map();

      //now open it for reading
      boost::shared_ptr<mat_t> mat = 
        io::detail::make_matfile(m_filename, m_mode);

      if (!mat) {
        boost::format f("uninitialized matlab file (%s) cannot be read");
        f % m_filename;
        throw std::runtime_error(f.str().c_str());
      }

      io::detail::read_array(mat, buffer);

    }

    virtual void arrayset_read(ca::interface& buffer, size_t index) {
      
      //do we need to reload the file?
      if (!m_type.is_valid()) try_reload_map();

      //now open it for reading
      boost::shared_ptr<mat_t> mat = 
        io::detail::make_matfile(m_filename, m_mode);

      if (!mat) {
        boost::format f("uninitialized matlab file (%s) cannot be read");
        f % m_filename;
        throw std::runtime_error(f.str().c_str());
      }

      io::detail::read_array(mat, buffer, (*m_map)[m_id[index]].first);

    }

    virtual size_t arrayset_append (const ca::interface& buffer) {

      //do we need to reload the file?
      if (!m_type.is_valid()) try_reload_map();

      //now open it for writing.
      boost::shared_ptr<mat_t> mat =
        io::detail::make_matfile(m_filename, m_mode);

      if (!mat) {
        boost::format f("cannot open matlab file at '%s' for writing");
        f % m_filename;
        throw std::runtime_error(f.str().c_str());
      }

      //checks typing is right
      if (m_type.is_valid() && !m_type.is_compatible(buffer.type())) {
        boost::format f("cannot append with different buffer type (%s) than the one already initialized (%s)");
        f % buffer.type().str() % m_type.str();
        throw std::invalid_argument(f.str().c_str());
      }

      //all is good at this point, just write it. 

      //choose variable name
      size_t next_index = 0;
      if (m_id.size()) next_index = *m_id.rbegin() + 1;
      std::ostringstream varname("array_");
      varname << next_index;

      io::detail::write_array(mat, varname.str(), buffer);

      mat.reset(); ///< force data flushing

      if (!m_type.is_valid()) try_reload_map();
      else {
        //optimization: don't reload the map, just update internal cache
        ++m_size;
        (*m_map)[next_index] = std::make_pair(varname.str(), buffer.type());
        m_id.push_back(next_index);
      }
      
      return m_size-1;
    }
    
    virtual void array_write (const ca::interface& buffer) {

      static std::string varname("array");

      //this file is supposed to hold a single array. delete it if it exists
      fs::path path (m_filename);
      if (fs::exists(m_filename)) fs::remove(m_filename);

      boost::shared_ptr<mat_t> mat = io::detail::make_matfile(m_filename, 
          m_mode);
      if (!mat) {
        boost::format f("cannot open matlab file at '%s' for writing");
        f % m_filename;
        throw std::runtime_error(f.str().c_str());
      }

      io::detail::write_array(mat, varname, buffer);

      mat.reset(); ///< forces data flushing (not really required here...)

      //updates internal map w/o looking to the output file.
      m_size = 1;
      (*m_map)[0] = std::make_pair(varname, buffer.type());
      m_id.push_back(0);

    }

  private: //representation

    typedef std::map<size_t, std::pair<std::string, ca::typeinfo> > map_type;

    std::string m_filename;
    enum mat_acc m_mode;
    boost::shared_ptr<map_type> m_map;
    ca::typeinfo m_type;
    size_t       m_size;
    std::vector<size_t> m_id;

    static std::string s_codecname;

};

std::string MatFile::s_codecname = "bob.matlab";

/**
 * From this point onwards we have the registration procedure. If you are
 * looking at this file for a coding example, just follow the procedure bellow,
 * minus local modifications you may need to apply.
 */

/**
 * This defines the factory method F that can create codecs of this type.
 * 
 * Here are the meanings of the mode flag that should be respected by your
 * factory implementation:
 *
 * 'r': opens for reading only - no modifications can occur; it is an
 *      error to open a file that does not exist for read-only operations.
 * 'w': opens for reading and writing, but truncates the file if it
 *      exists; it is not an error to open files that do not exist with
 *      this flag. 
 * 'a': opens for reading and writing - any type of modification can 
 *      occur. If the file does not exist, this flag is effectively like
 *      'w'.
 *
 * Returns a newly allocated File object that can read and write data to the
 * file using a specific backend.
 *
 * @note: This method can be static.
 */
static boost::shared_ptr<io::File> 
make_file (const std::string& path, char mode) {

  return boost::make_shared<MatFile>(path, mode);

}

/**
 * Takes care of codec registration per se.
 */
static bool register_codec() {

  boost::shared_ptr<io::CodecRegistry> instance =
    io::CodecRegistry::instance();
  
  instance->registerExtension(".mat", "Matlab binary files (v4 and superior)", &make_file);

  return true;

}

static bool codec_registered = register_codec();
