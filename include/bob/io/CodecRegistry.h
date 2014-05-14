/**
 * @file bob/io/CodecRegistry.h
 * @date Tue Oct 25 23:25:46 2011 +0200
 * @author Andre Anjos <andre.anjos@idiap.ch>
 *
 * @brief
 *
 * Copyright (C) 2011-2013 Idiap Research Institute, Martigny, Switzerland
 */

#ifndef BOB_IO_CODECREGISTRY_H
#define BOB_IO_CODECREGISTRY_H

#include <map>
#include <string>
#include <boost/shared_ptr.hpp>

#include <bob/io/File.h>

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

      void registerExtension(const char* extension, const char* description,
          file_factory_t factory);

      void deregisterFactory(file_factory_t factory);
      void deregisterExtension(const char* ext);

      /**
       * Returns the codec description, if an extension was registered with the
       * matching input string. Otherwise, returns 0.
       */
      const char* getDescription(const char* ext);

      file_factory_t findByExtension(const char* ext);
      file_factory_t findByFilenameExtension(const char* fn);

      bool isRegistered(const char* ext);

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
