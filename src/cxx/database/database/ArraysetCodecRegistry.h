/**
 * @file database/ArrayCodecRegistry.h
 * @author <a href="mailto:andre.anjos@idiap.ch">Andre Anjos</a> 
 *
 * @brief 
 */

#ifndef TORCH_DATABASE_ARRAYSETCODECREGISTRY_H 
#define TORCH_DATABASE_ARRAYSETCODECREGISTRY_H

#include <map>
#include <string>
#include <boost/shared_ptr.hpp>

#include "core/Exception.h"

#include "database/ArraysetCodec.h"

namespace Torch { namespace database {

  /**
   * The ArraysetCodecRegistry holds registered converters for different types
   * of input files. It manages registration and helps the user in picking the
   * best codecs for their data. This class is a singleton (single global
   * variable).
   */
  class ArraysetCodecRegistry {

    public:

      static void addCodec(boost::shared_ptr<Torch::database::ArraysetCodec> codec);

      static boost::shared_ptr<const Torch::database::ArraysetCodec>
        getCodecByName(const std::string& name);

      static boost::shared_ptr<const Torch::database::ArraysetCodec>
        getCodecByExtension(const std::string& filename);

    private:
      ArraysetCodecRegistry(): s_name2codec(), s_extension2codec() {}
      // Not implemented
      ArraysetCodecRegistry( const ArraysetCodecRegistry&);

      static boost::shared_ptr<ArraysetCodecRegistry> instance();
      static boost::shared_ptr<ArraysetCodecRegistry> s_instance;

      std::map<std::string, boost::shared_ptr<Torch::database::ArraysetCodec> > s_name2codec;
      std::map<std::string, boost::shared_ptr<Torch::database::ArraysetCodec> > s_extension2codec;
    
  };

}}

#endif /* TORCH_DATABASE_ARRAYSETCODECREGISTRY_H */

