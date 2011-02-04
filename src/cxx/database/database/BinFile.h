/**
 * @file src/cxx/database/database/BinFile.h
 * @author <a href="mailto:Laurent.El-Shafey@idiap.ch">Laurent El Shafey</a> 
 *
 * @brief This class can be used to load and store multiarrays from binary files.
 */

#ifndef TORCH5SPRO_DATABASE_BIN_FILE_H
#define TORCH5SPRO_DATABASE_BIN_FILE_H 1

#include "database/BinFileHeader.h"
#include "database/Dataset.h"
#include "core/StaticComplexCast.h"

namespace Torch {
  /**
   * \ingroup libdatabase_api
   * @{
   *
   */
  namespace database {

    /**
     * @brief Defines the flags that might be used when loading/storing
     * a file containing blitz arrays.
     */
    enum _BinFileFlag {
      _append  = 1L << 0,
      _in      = 1L << 3,
      _out     = 1L << 4
    };


    /**
     *  @brief This class can be used for loading and storing multiarrays
     *  from/to binary files
     */
    class BinFile
    {
      public:
        /**
         * @brief Define the bitmask type for providing information
         * about the type of the stream.
         */
        typedef _BinFileFlag openmode;
        static const openmode append  = _append;
        static const openmode in      = _in;
        static const openmode out     = _out; 

        /**
         * @brief Constructor
         */
        BinFile(const std::string& filename, openmode f);

        /**
         * @brief Destructor
         */
        ~BinFile();

        /**
         * Close the BinFile
         */
        void close();

        /** 
         * @brief Put a Blitz++ multiarray of a given type into the output
         * stream/file. If the type/shape have not yet been set, it is set 
         * according to the type and shape given in the blitz array, otherwise
         * the type/shape should match or an exception is thrown.
         */
        template <typename T, int D> void write(const blitz::Array<T,D>& bl);

        /**
         * @brief Save an Arrayset into a binary file
         */
        void write( boost::shared_ptr<const Arrayset> arrayset);

        /**
         * @brief Save an Array into a binary file
         */
        void write( boost::shared_ptr<const Array> array);

        /**
         * @brief Load one blitz++ multiarray from the input stream/file
         * All the multiarrays saved have the same dimensions.
         */
        template <typename T, int d> blitz::Array<T,d> read();
        template <typename T, int d> 
        blitz::Array<T,d> read(size_t index);

        /** 
         * @brief Load an Arrayset from a binary file
         */
        void read( boost::shared_ptr<Arrayset> arrayset);

        /** 
         * @brief Load an Array from a binary file
         */
        void read( boost::shared_ptr<Array> array);


        /**
         * @brief Get the Element type
         * @warning An exception is thrown if nothing was written so far
         */
        array::ElementType getElementType() const { 
          headerInitialized(); 
          return m_header.m_elem_type; 
        }
        /**
         * @brief Get the number of dimensions
         * @warning An exception is thrown if nothing was written so far
         */
        size_t getNDimensions() const {  
          headerInitialized(); 
          return m_header.m_n_dimensions; 
        }
        /**
         * @brief Get the shape of each array
         * @warning An exception is thrown if nothing was written so far
         */
        const size_t* getShape() const { 
          headerInitialized(); 
          return m_header.m_shape; 
        }
        /**
         * @brief Get the shape of each array in a blitz format
         * @warning An exception is thrown if nothing was written so far
         */
        template<int d>
        void getShape( blitz::TinyVector<int,d>& res ) const {
          headerInitialized(); 
          m_header.getShape(res);
        }
        /**
         * @brief Get the number of samples/arrays written so far
         * @warning An exception is thrown if nothing was written so far
         */
        size_t getNSamples() const { 
          headerInitialized(); 
          return m_n_arrays_written; 
        }
        /**
         * @brief Get the number of elements per array
         * @warning An exception is thrown if nothing was written so far
         */
        size_t getNElements() const { 
          headerInitialized(); 
          return m_header.m_n_elements; 
        }
        /**
         * @brief Get the size along a particular dimension
         * @warning An exception is thrown if nothing was written so far
         */
        size_t getSize(size_t dim_index) const { 
          headerInitialized(); 
          return m_header.getSize(dim_index); 
        }

        /**
         * @brief Initialize the binary file with the given type
         * and shape.
         */
        void initBinaryFile(const array::ElementType type,
          const size_t shape[array::N_MAX_DIMENSIONS_ARRAY]) {
            initHeader(type, shape);
        }


      private:
        /**
         * @brief Put a void C-style multiarray into the output stream/file
         * @warning This is the responsability of the user to check
         * the correctness of the type and size of the memory block 
         * pointed by the void pointer.
         */
        BinFile& write(const void* multi_array);

        /** 
         * @brief Put a C-style multiarray of a given type into the output
         * stream/file by casting it to the correct type.
         * @warning The C-style array has to be allocated with the proper 
         * dimensions.
         */
        template <typename T> 
        BinFile& writeWithCast(const T* multi_array);

        /**
         * @brief Write the data from a blitz array into a binary file
         * @warning It assumes that the shape and type match.
         */ 
        template <typename T, int D> 
        void writeBlitz(const blitz::Array<T,D>& bl) {
          error << "Unsupported type or number of dimensions" << std::endl;
          throw Exception();
        }
        
        /**
         * @brief Put a void C-style multiarray into the output stream/file
         * @warning This is the responsability of the user to check
         * the correctness of the type and size of the memory block 
         * pointed by the void pointer
         */
        BinFile& read(void* multi_array);

        /**
         * @brief Get one C-style array from the input stream/file, and cast
         * it to the given type.
         * @warning The C-style array has to be allocated with the proper 
         * dimensions
         */
        template <typename T> BinFile& readWithCast(T* multiarray);

        /**
         * @brief Check if the end of the binary file is reached
         */
        void endOfFile() {
          if(m_current_array >= m_header.m_n_samples ) {
            error << "The end of the binary file has been reached." << 
              std::endl;
            throw Exception();
          }
        }

        /**
         * @brief Check that the header has been initialized, and raise an
         * exception if not
         */
        void headerInitialized() const { 
          if(!m_header_init) {
            error << "The binary file has not yet been initialized." << std::endl;
            throw Exception();
          }
        }

        /**
         * @brief Initialize the header of the (output) stream with the given
         * type and shape
         */
        void initHeader(const array::ElementType type, 
            const size_t shape[array::N_MAX_DIMENSIONS_ARRAY]);

        /**
         * @brief Initialize the header with a blitz array
         */
        template <typename T, int D> 
        void initHeader(const blitz::Array<T,D>& bl);

        /**
         * @brief Initialize the part of the header which requires 
         * specialization with a blitz array
         */
        template <typename T, int D> 
        void initTypeHeader(const blitz::Array<T,D>& bl);


       /**
         * Attributes
         */
        bool m_header_init;
        size_t m_current_array;
        size_t m_n_arrays_written;
        std::fstream m_stream;
        BinFileHeader m_header;
        openmode m_openmode;
    };


#define WRITE_BLITZ_DECL(T,D) template <> \
  void BinFile::writeBlitz(const blitz::Array<T,D>& bl); \

        WRITE_BLITZ_DECL(bool,1)
        WRITE_BLITZ_DECL(bool,2)
        WRITE_BLITZ_DECL(bool,3)
        WRITE_BLITZ_DECL(bool,4)
        WRITE_BLITZ_DECL(int8_t,1)
        WRITE_BLITZ_DECL(int8_t,2)
        WRITE_BLITZ_DECL(int8_t,3)
        WRITE_BLITZ_DECL(int8_t,4)
        WRITE_BLITZ_DECL(int16_t,1)
        WRITE_BLITZ_DECL(int16_t,2)
        WRITE_BLITZ_DECL(int16_t,3)
        WRITE_BLITZ_DECL(int16_t,4)
        WRITE_BLITZ_DECL(int32_t,1)
        WRITE_BLITZ_DECL(int32_t,2)
        WRITE_BLITZ_DECL(int32_t,3)
        WRITE_BLITZ_DECL(int32_t,4)
        WRITE_BLITZ_DECL(int64_t,1)
        WRITE_BLITZ_DECL(int64_t,2)
        WRITE_BLITZ_DECL(int64_t,3)
        WRITE_BLITZ_DECL(int64_t,4)
        WRITE_BLITZ_DECL(uint8_t,1)
        WRITE_BLITZ_DECL(uint8_t,2)
        WRITE_BLITZ_DECL(uint8_t,3)
        WRITE_BLITZ_DECL(uint8_t,4)
        WRITE_BLITZ_DECL(uint16_t,1)
        WRITE_BLITZ_DECL(uint16_t,2)
        WRITE_BLITZ_DECL(uint16_t,3)
        WRITE_BLITZ_DECL(uint16_t,4)
        WRITE_BLITZ_DECL(uint32_t,1)
        WRITE_BLITZ_DECL(uint32_t,2)
        WRITE_BLITZ_DECL(uint32_t,3)
        WRITE_BLITZ_DECL(uint32_t,4)
        WRITE_BLITZ_DECL(uint64_t,1)
        WRITE_BLITZ_DECL(uint64_t,2)
        WRITE_BLITZ_DECL(uint64_t,3)
        WRITE_BLITZ_DECL(uint64_t,4)
        WRITE_BLITZ_DECL(float,1)
        WRITE_BLITZ_DECL(float,2)
        WRITE_BLITZ_DECL(float,3)
        WRITE_BLITZ_DECL(float,4)
        WRITE_BLITZ_DECL(double,1)
        WRITE_BLITZ_DECL(double,2)
        WRITE_BLITZ_DECL(double,3)
        WRITE_BLITZ_DECL(double,4)
        WRITE_BLITZ_DECL(std::complex<float>,1)
        WRITE_BLITZ_DECL(std::complex<float>,2)
        WRITE_BLITZ_DECL(std::complex<float>,3)
        WRITE_BLITZ_DECL(std::complex<float>,4)
        WRITE_BLITZ_DECL(std::complex<double>,1)
        WRITE_BLITZ_DECL(std::complex<double>,2)
        WRITE_BLITZ_DECL(std::complex<double>,3)
        WRITE_BLITZ_DECL(std::complex<double>,4)

#define INIT_HEADER_DECL(T,D) template<> \
  void BinFile::initTypeHeader(const blitz::Array<T,D>& bl); \

        INIT_HEADER_DECL(bool,1)
        INIT_HEADER_DECL(bool,2)
        INIT_HEADER_DECL(bool,3)
        INIT_HEADER_DECL(bool,4)
        INIT_HEADER_DECL(int8_t,1)
        INIT_HEADER_DECL(int8_t,2)
        INIT_HEADER_DECL(int8_t,3)
        INIT_HEADER_DECL(int8_t,4)
        INIT_HEADER_DECL(int16_t,1)
        INIT_HEADER_DECL(int16_t,2)
        INIT_HEADER_DECL(int16_t,3)
        INIT_HEADER_DECL(int16_t,4)
        INIT_HEADER_DECL(int32_t,1)
        INIT_HEADER_DECL(int32_t,2)
        INIT_HEADER_DECL(int32_t,3)
        INIT_HEADER_DECL(int32_t,4)
        INIT_HEADER_DECL(int64_t,1)
        INIT_HEADER_DECL(int64_t,2)
        INIT_HEADER_DECL(int64_t,3)
        INIT_HEADER_DECL(int64_t,4)
        INIT_HEADER_DECL(uint8_t,1)
        INIT_HEADER_DECL(uint8_t,2)
        INIT_HEADER_DECL(uint8_t,3)
        INIT_HEADER_DECL(uint8_t,4)
        INIT_HEADER_DECL(uint16_t,1)
        INIT_HEADER_DECL(uint16_t,2)
        INIT_HEADER_DECL(uint16_t,3)
        INIT_HEADER_DECL(uint16_t,4)
        INIT_HEADER_DECL(uint32_t,1)
        INIT_HEADER_DECL(uint32_t,2)
        INIT_HEADER_DECL(uint32_t,3)
        INIT_HEADER_DECL(uint32_t,4)
        INIT_HEADER_DECL(uint64_t,1)
        INIT_HEADER_DECL(uint64_t,2)
        INIT_HEADER_DECL(uint64_t,3)
        INIT_HEADER_DECL(uint64_t,4)
        INIT_HEADER_DECL(float,1)
        INIT_HEADER_DECL(float,2)
        INIT_HEADER_DECL(float,3)
        INIT_HEADER_DECL(float,4)
        INIT_HEADER_DECL(double,1)
        INIT_HEADER_DECL(double,2)
        INIT_HEADER_DECL(double,3)
        INIT_HEADER_DECL(double,4)
        INIT_HEADER_DECL(std::complex<float>,1)
        INIT_HEADER_DECL(std::complex<float>,2)
        INIT_HEADER_DECL(std::complex<float>,3)
        INIT_HEADER_DECL(std::complex<float>,4)
        INIT_HEADER_DECL(std::complex<double>,1)
        INIT_HEADER_DECL(std::complex<double>,2)
        INIT_HEADER_DECL(std::complex<double>,3)
        INIT_HEADER_DECL(std::complex<double>,4)


  }
  /**
   * @}
   */
}

#include "database/BinFile.cc"

#endif /* TORCH5SPRO_DATABASE_BIN_FILE_H */

