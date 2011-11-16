/**
 * @file io/CodecRegistry.h
 * @author <a href="mailto:andre.anjos@idiap.ch">Andre Anjos</a> 
 *
 * @brief 
 */

#ifndef TORCH_IO_CODECREGISTRY_H 
#define TORCH_IO_CODECREGISTRY_H

#include <map>
#include <string>
#include <boost/shared_ptr.hpp>

#include "io/File.h"

namespace Torch { namespace io {

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
  boost::shared_ptr<File> open
    (const std::string& filename, const std::string& pretend_extension, 
     char mode);

}}

#endif /* TORCH_IO_CODECREGISTRY_H */

