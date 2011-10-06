/**
 * @author Andre Anjos <andre.anjos@idiap.ch>
 * @date Thu  6 Oct 09:10:42 2011 CEST
 *
 * @brief This class describes how an array will load and save data from an
 * external file.
 */

#ifndef TORCH_IO_FILEARRAY_H 
#define TORCH_IO_FILEARRAY_H

#include <string>

#include <blitz/array.h>
#include <boost/shared_ptr.hpp>

#include "core/array_type.h"

#include "io/buffer.h"
#include "io/ArrayCodec.h"

namespace Torch { namespace io {

  /**
   * Describes information on the file, codec and the type information that
   * could be retrieved from that.
   */
  struct fileinfo {

    std::string filename;
    boost::shared_ptr<const Torch::io::ArrayCodec> codec;

    /**
     * Default constructor -- empty
     */
    fileinfo() { }

    /**
     * Builds a new file info
     */
    fileinfo(const std::string& filename, const std::string& codecname);

    /**
     * Copy construct
     */
    fileinfo(const fileinfo& other);

    /**
     * Reloads the file specifications to a given type information
     */
    void read_type(typeinfo& info);

  };

  /** 
   * A filearray represents an array to which the data is stored on an external
   * file. It has no notion of management aspects of arrays. It just knows how
   * to handle its data. Reading and writing data to a filearray will read and
   * write data to/from a file. This operation can be costly.
   */
  class filearray {

    public:

      /**
       * Specify the filename and the code to read it. If you don't specify the
       * codec, I'll try to guess it from the registry. If the user gives a
       * relative path, it is relative to the current working directory. The
       * last parameter should be used in the case this is supposed to create a
       * new file, in which case the specifications will not be pre-loaded.
       */
      filearray(const std::string& filename, const std::string&
          codecname="", bool newfile=false);

      /**
       * Destroys an array.
       */
      virtual ~filearray();

      /**
       * Loads the data from the file. If the passed buffer object does not
       * contain enough space for the object, a re-allocation will happen.
       */
      void load(buffer& array) const;

      /**
       * Saves data in the file. Please note that blitz::Array<>'s will be
       * implicetly converted to InlinedArrayImpl as required.
       */
      void save(const buffer& data);

      /**
       * Sets the filename of where the data is contained, re-writing the data
       * if the codec changes. Otherwise, just move.
       */
      void move(const std::string& filename, const std::string& codecname="");

      /**
       * Gets a descriptive information.
       */
      inline const fileinfo& info() const { return m_info; }

      /**
       * Some information about the type
       */
      inline const typeinfo& type() const { return m_type; }

    private: //representation
      fileinfo m_info;
      typeinfo m_type;

  };

}}

#endif /* TORCH_IO_FILEARRAY_H */
