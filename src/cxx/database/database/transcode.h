/**
 * @author <a href="mailto:andre.anjos@idiap.ch">Andre Anjos</a> 
 *
 * @brief A set of non-member methods to transcode files.
 */

#ifndef TORCH_DATABASE_TRANSCODE_H 
#define TORCH_DATABASE_TRANSCODE_H

#include <string>

namespace Torch { namespace database {

  /**
   * Transcodes a file containing a single array to another file, possibly in a
   * different format. The codecs to be used are derived from the filename
   * extensions.
   *
   * In the event of readout errors or unsupported data types, exceptions will
   * be thrown.
   */
  void array_transcode (const std::string& from, const std::string& to);

  /**
   * Transcodes a file containing a single array to another file, possibly in a
   * different format. The codecs are defined by the codec names given as
   * parameter.
   *
   * In the event of readout errors or unsupported data types, exceptions will
   * be thrown.
   */
  void array_transcode (const std::string& from, 
                        const std::string& from_codecname,
                        const std::string& to, 
                        const std::string& to_codecname);

  /*
   * Transcodes a file containing a arrayset to another file, possibly in a
   * different format. The codecs to be used are derived from the filename
   * extensions.
   *
   * In the event of readout errors or unsupported data types, exceptions will
   * be thrown.
   */
  void arrayset_transcode (const std::string& from, const std::string& to);

  /**
   * Transcodes a file containing a arrayset to another file, possibly in a
   * different format. The codecs are defined by the codec names given as
   * parameter.
   *
   * In the event of readout errors or unsupported data types, exceptions will
   * be thrown.
   */
  void arrayset_transcode (const std::string& from, 
                           const std::string& from_codecname,
                           const std::string& to, 
                           const std::string& to_codecname);

}}

#endif /* TORCH_DATABASE_TRANSCODE_H */

