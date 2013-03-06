/**
 * @file bob/io/CodecRegistry.h
 * @date Tue Oct 25 23:25:46 2011 +0200
 * @author Andre Anjos <andre.anjos@idiap.ch>
 *
 * @brief
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
 
      /**
       * Sets and unsets double-registration ignore flag
       */
      static bool ignoreDoubleRegistration() { return instance()->s_ignore; }
      static void ignoreDoubleRegistration(bool v) { instance()->s_ignore = v; }

    public: //object access

      void registerExtension(const std::string& extension,
          const std::string& description,
          file_factory_t factory);

      void deregisterFactory(file_factory_t factory);
      void deregisterExtension(const std::string& codecname);

      file_factory_t findByExtension(const std::string& ext);
      file_factory_t findByFilenameExtension(const std::string& fn);

      bool isRegistered(const std::string& ext);

    private:

      CodecRegistry(): s_extension2codec(), s_ignore(false) {}

      // Not implemented
      CodecRegistry( const CodecRegistry&);

      std::map<std::string, file_factory_t> s_extension2codec;
      std::map<std::string, std::string> s_extension2description;
      bool s_ignore; ///< shall I ignore double-registrations?
    
  };

}}

#endif /* BOB_IO_CODECREGISTRY_H */
