/**
 * @file src/core/core/BinOutputFile.h
 * @author <a href="mailto:Laurent.El-Shafey@idiap.ch">Laurent El Shafey</a> 
 *
 * @brief This class can be used to store multiarrays into files.
 */

#ifndef TORCH5SPRO_CORE_BIN_OUTPUT_FILE_H
#define TORCH5SPRO_CORE_BIN_OUTPUT_FILE_H

#include "core/BinFileHeader.h"
#include "core/Dataset2.h"
#include <string>

namespace Torch {
  /**
   * \ingroup libcore_api
   * @{
   *
   */
  namespace core {

      /**
       *  @brief the OutputFile class for storing multiarrays into files
       */
      class BinOutputFile
      {
        public:
          /**
           * Constructor
           */
          BinOutputFile(const std::string& filename, bool append = false);

          /**
           * Destructor
           */
          virtual ~BinOutputFile();

          /**
           * Initialize the header of the output stream with the given type
           * and shape
           */
          void initHeader(const array::ArrayType type, const size_t *shape);

          /**
           * Close the BlitzOutputFile and update the header
           */
          void close();


          /**
           * @brief Put a C-style multiarray into the output stream/file
           * @warning This is the responsability of the user to check
           * the correctness of the type and size of the memory block 
           * pointed by the void pointer
           */
          BinOutputFile& operator<<(const void* multi_array);

          /**
           * @brief Save an Arrayset into a binary file
           */
          void save(const Arrayset& arrayset);

          /**
           * @brief Save an Array into a binary file
           */
          void save(const Array& array);

          /**
           * Return the header
           */
          BinFileHeader& getHeader() { return m_header; }

        private:
          void checkWriteInit();

          bool m_write_init;
          std::fstream m_out_stream;
          BinFileHeader m_header;
          size_t m_n_arrays_written;
      };


  }
}

#endif /* TORCH5SPRO_CORE_BIN_OUTPUT_FILE_H */

