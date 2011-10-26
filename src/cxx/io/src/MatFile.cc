/**
 * @author <a href="mailto:andre.dos.anjos@gmail.com">Andre Anjos</a> 
 * @date Sun 20 Feb 09:51:30 2011 
 *
 * @brief Implements the matlab (.mat) array codec using matio
 */

#include <boost/make_shared.hpp>
#include <boost/filesystem.hpp>
#include <algorithm>

#include "io/MatUtils.h"
#include "io/CodecRegistry.h"
#include "io/Exception.h"

namespace fs = boost::filesystem;
namespace io = Torch::io;

class MatFile: public io::File {

  public: //api

    MatFile(const std::string& path, char mode):
      m_filename(path),
      m_mode( (mode=='r')? MAT_ACC_RDONLY : MAT_ACC_RDWR ),
      m_size(0) {
        if (mode == 'r' || (mode == 'a' && fs::exists(path))) reload_map();
      }

    virtual ~MatFile() { }

    void reload_map () {
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
        throw io::DimensionError(m_type.nd, TORCH_MAX_DIM);
      if (m_type.dtype == Torch::core::array::t_unknown) 
        throw io::UnsupportedTypeError(m_type.dtype);
    }

    virtual const std::string& filename() const {
      return m_filename;
    }

    virtual const io::typeinfo& array_type () const {
      return m_type;
    }

    virtual const io::typeinfo& arrayset_type () const {
      return m_type;
    }

    virtual size_t arrayset_size() const {
      return m_size;
    }

    virtual const std::string& name() const {
      return s_codecname;
    }

    virtual void array_read(io::buffer& buffer) {

      boost::shared_ptr<mat_t> mat = 
        io::detail::make_matfile(m_filename, m_mode);

      if (!mat)
        throw std::runtime_error("uninitialized matlab file cannot be read");

      //do we need to reload the file?
      if (!m_type.is_valid()) reload_map();

      io::detail::read_array(mat, buffer);

    }

    virtual void arrayset_read(io::buffer& buffer, size_t index) {

      boost::shared_ptr<mat_t> mat = 
        io::detail::make_matfile(m_filename, m_mode);

      if (!mat)
        throw std::runtime_error("uninitialized matlab file cannot be read");

      //do we need to reload the file?
      if (!m_type.is_valid()) reload_map();

      io::detail::read_array(mat, buffer, (*m_map)[m_id[index]].first);

    }

    virtual size_t arrayset_append (const io::buffer& buffer) {

      boost::shared_ptr<mat_t> mat =
        io::detail::make_matfile(m_filename, m_mode);

      if (!mat)
        throw std::runtime_error("cannot open matlab file for writing");

      //do we need to reload the file?
      if (!m_type.is_valid()) reload_map();

      //checks typing is right
      if (m_type.is_valid() && !m_type.is_compatible(buffer.type()))
        throw std::invalid_argument("cannot append with different buffer type than the one already initialized");

      //all is good at this point, just write it. 

      //choose variable name
      size_t next_index = 0;
      if (m_id.size()) next_index = *m_id.rbegin() + 1;
      std::ostringstream varname("array_");
      varname << next_index;

      io::detail::write_array(mat, varname.str(), buffer);

      mat.reset(); ///< force data flushing

      if (!m_type.is_valid()) reload_map();
      else {
        //optimization: don't reload the map, just update internal cache
        ++m_size;
        (*m_map)[next_index] = std::make_pair(varname.str(), buffer.type());
        m_id.push_back(next_index);
      }
      
      return m_size-1;
    }
    
    virtual void array_write (const io::buffer& buffer) {

      static std::string varname("array");

      //this file is supposed to hold a single array. delete it if it exists
      fs::path path (m_filename);
      if (fs::exists(m_filename)) fs::remove(m_filename);

      boost::shared_ptr<mat_t> mat = io::detail::make_matfile(m_filename, m_mode);
      if (!mat)
        throw std::runtime_error("cannot open matlab file for writing");

      io::detail::write_array(mat, varname, buffer);

      mat.reset(); ///< forces data flushing (not really required here...)

      //updates internal map w/o looking to the output file.
      m_size = 1;
      (*m_map)[0] = std::make_pair(varname, buffer.type());
      m_id.push_back(0);

    }

  private: //representation

    typedef std::map<size_t, std::pair<std::string, io::typeinfo> > map_type;

    std::string m_filename;
    enum mat_acc m_mode;
    boost::shared_ptr<map_type> m_map;
    io::typeinfo m_type;
    size_t       m_size;
    std::vector<size_t> m_id;

    static std::string s_codecname;

};

std::string MatFile::s_codecname = "torch.matlab";

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
  
  instance->registerExtension(".mat", &make_file);

  return true;

}

static bool codec_registered = register_codec();
