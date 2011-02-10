/**
 * @file database/ArrayCodecRegistry.h
 * @author <a href="mailto:andre.anjos@idiap.ch">Andre Anjos</a> 
 *
 * @brief 
 */

#ifndef TORCH_DATABASE_ARRAYCODECREGISTRY_H 
#define TORCH_DATABASE_ARRAYCODECREGISTRY_H

#include <map>
#include <string>
#include <boost/shared_ptr.hpp>

#include "core/Exception.h"

#include "database/ArrayCodec.h"

namespace Torch { namespace database {

  /**
   * The ArrayCodecRegistry holds registered converters for different types of
   * input files. It manages registration and helps the user in picking the
   * best codecs for their data. This class is a singleton (single global
   * variable).
   */
  class ArrayCodecRegistry {

    public:

      static void addCodec(boost::shared_ptr<Torch::database::ArrayCodec> codec);

      static boost::shared_ptr<const Torch::database::ArrayCodec>
        getCodecByName(const std::string& name);

      static boost::shared_ptr<const Torch::database::ArrayCodec>
        getCodecByExtension(const std::string& filename);

    private:
      ArrayCodecRegistry(): s_name2codec(), s_extension2codec() {}
      // Not implemented
      ArrayCodecRegistry( const ArrayCodecRegistry&);

      static boost::shared_ptr<ArrayCodecRegistry> instance();

      std::map<std::string, boost::shared_ptr<Torch::database::ArrayCodec> > s_name2codec;
      std::map<std::string, boost::shared_ptr<Torch::database::ArrayCodec> > s_extension2codec;
    
  };

}}

#endif /* TORCH_DATABASE_ARRAYCODECREGISTRY_H */

