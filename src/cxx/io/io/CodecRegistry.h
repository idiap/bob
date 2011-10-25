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

      template <typename T>
      static void getExtensions (T& t) {
        boost::shared_ptr<CodecRegistry> ptr = instance();
        for (std::map<std::string, file_factory_t>::const_iterator it = ptr->s_extension2codec.begin(); it != ptr->s_extension2codec.end(); ++it) t.push_back(it->first);
      }

    public: //object access

      void registerExtension(const std::string& extension,
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

