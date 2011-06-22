/**
 * @file io/ArrayCodecRegistry.h
 * @author <a href="mailto:andre.anjos@idiap.ch">Andre Anjos</a> 
 *
 * @brief 
 */

#ifndef TORCH_IO_ARRAYCODECREGISTRY_H 
#define TORCH_IO_ARRAYCODECREGISTRY_H

#include <map>
#include <string>
#include <boost/shared_ptr.hpp>

#include "core/Exception.h"

#include "io/ArrayCodec.h"

namespace Torch { namespace io {

  /**
   * The ArrayCodecRegistry holds registered converters for different types of
   * input files. It manages registration and helps the user in picking the
   * best codecs for their data. This class is a singleton (single global
   * variable).
   */
  class ArrayCodecRegistry {

    public:

      static void addCodec(boost::shared_ptr<Torch::io::ArrayCodec> codec);
      static void removeCodecByName(const std::string& codecname);

      static boost::shared_ptr<const Torch::io::ArrayCodec>
        getCodecByName(const std::string& name);

      static boost::shared_ptr<const Torch::io::ArrayCodec>
        getCodecByExtension(const std::string& filename);

      template <typename T>
      static void getCodecNames(T& t) {
        boost::shared_ptr<ArrayCodecRegistry> ptr = instance();
        for (std::map<std::string, boost::shared_ptr<Torch::io::ArrayCodec> >::const_iterator it = ptr->s_name2codec.begin(); it != ptr->s_name2codec.end(); ++it) t.push_back(it->first);
      }

      template <typename T>
      static void getExtensions(T& t) {
        boost::shared_ptr<ArrayCodecRegistry> ptr = instance();
        for (std::map<std::string, boost::shared_ptr<Torch::io::ArrayCodec> >::const_iterator it = ptr->s_extension2codec.begin(); it != ptr->s_extension2codec.end(); ++it) t.push_back(it->first);
      }

    private:
      ArrayCodecRegistry(): s_name2codec(), s_extension2codec() {}
      // Not implemented
      ArrayCodecRegistry( const ArrayCodecRegistry&);

      static boost::shared_ptr<ArrayCodecRegistry> instance();

      std::map<std::string, boost::shared_ptr<Torch::io::ArrayCodec> > s_name2codec;
      std::map<std::string, boost::shared_ptr<Torch::io::ArrayCodec> > s_extension2codec;
    
  };

}}

#endif /* TORCH_IO_ARRAYCODECREGISTRY_H */

