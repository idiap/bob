/**
 * @file io/cxx/CSVFile.cc
 * @date Thu 10 May 2012 15:19:24 CEST
 * @author Andre Anjos <andre.anjos@idiap.ch>
 *
 * @brief Code to read and write CSV files to/from arrays. CSV files are always
 * treated as containing sequences of double precision numbers.
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

#include <sstream>
#include <fstream>
#include <string>
#include <boost/filesystem.hpp>
#include <boost/format.hpp>
#include <boost/filesystem.hpp>
#include <boost/make_shared.hpp>
#include <boost/tokenizer.hpp>

#include <boost/shared_array.hpp>
#include <boost/algorithm/string.hpp>

#include <bob/io/CodecRegistry.h>
#include <bob/io/Exception.h>

typedef boost::tokenizer<boost::escaped_list_separator<char> > Tokenizer;

class CSVFile: public bob::io::File {

  public: //api

    /**
     * Peeks the file contents for a type. We assume the element type to be
     * always doubles. This method, effectively, only peaks for the total
     * number of lines and the number of columns in the file.
     */
    void peek() {

      std::string line;
      size_t line_number = 0;
      size_t entries = 0;
      std::streampos cur_pos = 0;

      m_file.seekg(0); //< returns to the begin of file and start reading...

      while (std::getline(m_file,line)) {
        ++line_number;
        m_pos.push_back(cur_pos);
        cur_pos = m_file.tellg();
        Tokenizer tok(line);
        size_t size = std::distance(tok.begin(), tok.end());
        if (!entries) entries = size;
        else if (entries != size) {
          boost::format m("line %d at file '%s' contains %d entries instead of %d (expected)");
          m % line_number % m_filename % size % entries;
          throw std::runtime_error(m.str());
        }
      }

      if (!line_number) {
        m_newfile = true;
        m_pos.clear();
        return;
      }

      m_arrayset_type.dtype = bob::core::array::t_float64;
      m_arrayset_type.nd = 1;
      m_arrayset_type.shape[0] = entries;
      m_arrayset_type.update_strides();

      m_array_type = m_arrayset_type;
      m_array_type.nd = 2;
      m_array_type.shape[0] = m_pos.size();
      m_array_type.shape[1] = entries;
      m_array_type.update_strides();
    }

    CSVFile(const std::string& path, char mode):
      m_filename(path),
      m_newfile(false) {

        if (mode == 'r' || (mode == 'a' && boost::filesystem::exists(path))) { //try peeking
          
          if (mode == 'r') 
            m_file.open(m_filename.c_str(), std::ios::in);
          else if (mode == 'a')
            m_file.open(m_filename.c_str(), std::ios::app|std::ios::in|std::ios::out);
          if (!m_file.is_open()) {
            boost::format m("cannot open file '%s' for reading or appending");
            m % path;
            throw std::runtime_error(m.str());
          }

          peek(); ///< peek file properties
        }
        else {
          m_file.open(m_filename.c_str(), std::ios::trunc|std::ios::in|std::ios::out);
          
          if (!m_file.is_open()) {
            boost::format m("cannot open file '%s' for writing");
            m % path;
            throw std::runtime_error(m.str());
          }

          m_newfile = true;
        }

        //general precision settings, in case output is needed...
        m_file.precision(10);
        m_file.setf(std::ios_base::scientific, std::ios_base::floatfield);

      }

    virtual ~CSVFile() { }

    virtual const std::string& filename() const {
      return m_filename;
    }

    virtual const bob::core::array::typeinfo& type() const {
      return m_arrayset_type;
    }

    virtual const bob::core::array::typeinfo& type_all() const {
      return m_array_type;
    }

    virtual size_t size() const {
      return m_pos.size();
    }

    virtual const std::string& name() const {
      return s_codecname;
    }

    virtual void read_all(bob::core::array::interface& buffer) {
      if (m_newfile)
        throw std::runtime_error("uninitialized CSV file cannot be read");

      if (!buffer.type().is_compatible(m_array_type)) buffer.set(m_array_type);

      //read contents
      std::string line;
      if (m_file.eof()) m_file.clear(); ///< clear current "end" state.
      m_file.seekg(0);
      double* p = static_cast<double*>(buffer.ptr());
      while (std::getline(m_file, line)) {
        Tokenizer tok(line);
        for(Tokenizer::iterator k=tok.begin(); k!=tok.end(); ++k) {
          std::istringstream(*k) >> *(p++);
        }
      }
    }

    virtual void read(bob::core::array::interface& buffer, size_t index) {

      if (m_newfile)
        throw std::runtime_error("uninitialized CSV file cannot be read");

      if (!buffer.type().is_compatible(m_arrayset_type)) 
        buffer.set(m_arrayset_type);

      if (index >= m_pos.size()) {
        boost::format m("cannot array at position %d -- there is only %d entries at file '%s'");
        m % index % m_pos.size() % m_filename;
        throw std::runtime_error(m.str());
      }

      //reads a specific line from the file.
      std::string line;
      if (m_file.eof()) m_file.clear(); ///< clear current "end" state.
      m_file.seekg(m_pos[index]);
      if (!std::getline(m_file, line)) {
        boost::format m("could not seek to line %u (offset %u) while reading file '%s'");
        m % index % m_pos[index] % m_filename;
        throw std::runtime_error(m.str());
      }
      Tokenizer tok(line);
      double* p = static_cast<double*>(buffer.ptr());
      for(Tokenizer::iterator k=tok.begin(); k!=tok.end(); ++k) {
        std::istringstream(*k) >> *(p++);
      }

    }

    virtual size_t append (const bob::core::array::interface& buffer) {

      const bob::core::array::typeinfo& type = buffer.type();

      if (m_newfile) {
        if (type.nd != 1 || type.dtype != bob::core::array::t_float64) {
          boost::format m("cannot append %s to file '%s' - CSV files only accept 1D double precision float arrays");
          m % type.str() % m_filename;
          throw std::runtime_error(m.str());
        }
        m_pos.clear();
        m_arrayset_type = m_array_type = type;
        m_array_type.shape[1] = m_arrayset_type.shape[0];
        m_newfile = false;
      }

      else {

        //check compatibility
        if (!m_arrayset_type.is_compatible(buffer.type())) {
          boost::format m("CSV file '%s' only accepts arrays of type %s");
          m % m_filename % m_arrayset_type.str();
          throw std::runtime_error(m.str());
        }

      }

      const double* p = static_cast<const double*>(buffer.ptr());
      if (m_pos.size()) m_file << std::endl; ///< adds a new line
      m_pos.push_back(m_file.tellp()); ///< register start of line
      for (size_t k=1; k<type.shape[0]; ++k) m_file << *(p++) << ",";
      m_file << *(p++);
      m_array_type.shape[0] = m_pos.size();
      m_array_type.update_strides();
      return (m_pos.size()-1);

    }

    virtual void write (const bob::core::array::interface& buffer) {

      const bob::core::array::typeinfo& type = buffer.type();

      if (m_newfile) {
        if (type.nd != 2 || type.dtype != bob::core::array::t_float64) {
          boost::format m("cannot write %s to file '%s' - CSV files only accept a single 2D double precision float array as input");
          m % type.str() % m_filename;
          throw std::runtime_error(m.str());
        }
        const double* p = static_cast<const double*>(buffer.ptr());
        for (size_t l=1; l<type.shape[0]; ++l) {
          m_pos.push_back(m_file.tellp());
          for (size_t k=1; k<type.shape[1]; ++k) m_file << *(p++) << ",";
          m_file << *(p++) << std::endl;
        }
        for (size_t k=1; k<type.shape[1]; ++k) m_file << *(p++) << ",";
        m_file << *(p++);
        m_arrayset_type = type;
        m_arrayset_type.nd = 1;
        m_arrayset_type.shape[0] = type.shape[1];
        m_arrayset_type.update_strides();
        m_array_type = type;
        m_newfile = false;
        return;
      }

      //TODO
      throw std::runtime_error("Writing a 2D array to a CSV file that already contains entries is not implemented at the moment");

    }

  private: //representation
    std::fstream m_file;
    std::string m_filename;
    bool m_newfile;
    bob::core::array::typeinfo m_array_type;
    bob::core::array::typeinfo m_arrayset_type;
    std::vector<std::streampos> m_pos; ///< dictionary of line starts

    static std::string s_codecname;

};

std::string CSVFile::s_codecname = "bob.csv";

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
static boost::shared_ptr<bob::io::File> 
make_file (const std::string& path, char mode) {
  return boost::make_shared<CSVFile>(path, mode);

}

/**
 * Takes care of codec registration per se.
 */
static bool register_codec() {

  boost::shared_ptr<bob::io::CodecRegistry> instance =
    bob::io::CodecRegistry::instance();
  
  instance->registerExtension(".csv", "Comma-Separated Values", &make_file);
  instance->registerExtension(".txt", "Comma-Separated Values", &make_file);

  return true;

}

static bool codec_registered = register_codec();
