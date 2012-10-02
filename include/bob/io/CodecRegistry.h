/**
 * @file bob/io/CodecRegistry.h
 * @date Tue Oct 25 23:25:46 2011 +0200
 * @author Andre Anjos <andre.anjos@idiap.ch>
 *
 * @brief
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

#ifndef BOB_IO_CODECREGISTRY_H 
#define BOB_IO_CODECREGISTRY_H

#include <map>
#include <string>
#include <boost/shared_ptr.hpp>

#include "bob/io/File.h"

namespace bob { namespace io {

  /**
   * The CodecRegistry holds registered converters for different types of
   * input files. It manages registration and helps the user in picking the
   * best codecs for their data. This class is a singleton (single global
   * variable).
   */
  class CodecRegistry {

    public: //static access
      
      /**
       * Returns the singleton
       */
      static boost::shared_ptr<CodecRegistry> instance();

      static const std::map<std::string, std::string>& getExtensions () {
        boost::shared_ptr<CodecRegistry> ptr = instance();
        return ptr->s_extension2description;
      }

    public: //object access

      void registerExtension(const std::string& extension,
          const std::string& description,
          file_factory_t factory);

      void deregisterFactory(file_factory_t factory);
      void deregisterExtension(const std::string& codecname);

      file_factory_t findByExtension(const std::string& ext);
      file_factory_t findByFilenameExtension(const std::string& fn);

    private:

      CodecRegistry(): s_extension2codec() {}

      // Not implemented
      CodecRegistry( const CodecRegistry&);

      std::map<std::string, file_factory_t> s_extension2codec;
      std::map<std::string, std::string> s_extension2description;
    
  };

  /**
   * Creates a new array codec using the given "pretend" extension or the
   * filename extension itself if that is empty. The opening mode is passed
   * to the underlying registered File implementation.
   */
  boost::shared_ptr<File> open (const std::string& filename, char mode);

  /**
   * Opens the file pretending it has a different extension (that is, using a
   * different codec) then the one expected (if any). This allows you to write
   * a file with the extension you want, but still using one of the available
   * codecs.
   */
  boost::shared_ptr<File> open (const std::string& filename, char mode, 
      const std::string& pretend_extension);

  /**
   * Opens for reading and load all contents
   */
  template <typename T, int N> blitz::Array<T,N> load (const std::string& filename) {
    return open(filename, 'r')->read_all<T,N>();
  }

  /**
   * Opens for writing and write array to it
   */
  template <typename T, int N> void save (const std::string& filename, const blitz::Array<T,N>& data) {
    open(filename, 'w')->write(data);
  }

}}

#endif /* BOB_IO_CODECREGISTRY_H */
