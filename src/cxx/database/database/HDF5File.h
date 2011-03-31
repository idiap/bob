/**
 * @author <a href="mailto:andre.anjos@idiap.ch">Andre Anjos</a> 
 * @author <a href="mailto:Laurent.El-Shafey@idiap.ch">Laurent El Shafey</a> 
 * @date Wed 30 Mar 21:06:28 2011 
 *
 * @brief Torch support for HDF5 files. HDF5 is a open standard for
 * self-describing data files. You can get more information in this webpage:
 * http://www.hdfgroup.org/HDF5
 */

#ifndef TORCH_DATABASE_HDF5FILE_H 
#define TORCH_DATABASE_HDF5FILE_H

#include <map>
#include <vector>
#include <boost/filesystem.hpp>
#include <blitz/array.h>
#include <H5Cpp.h>

#include "core/array_common.h"
#include "database/Exception.h"

namespace Torch { namespace database {

  /**
   * This class allows users to read and write simple variables in HDF5 files.
   * Supported types are:
   *
   * 1) scalars and arrays of
   *    - bool
   *    - int8, 16, 32 and 64
   *    - uint8, 16, 32 and 64
   *    - float32 (float), 64 (double) and 128 (long double)
   *    - complex64 (2x float), 128 (2x double), 256 (2x long double)
   * 2) character scalars
   * 3) variable length string scalars
   *
   * The user can create a hierarchy of groups (sub-directories). For more
   * information about HDF5, please consult the consortium webpage: 
   * http://www.hdfgroup.org/HDF5
   */
  class HDF5File {

    public:

      /**
       * This enumeration defines the different values with which you can open
       * the files with
       */
      typedef enum mode_t {
        excl = 4, //H5F_ACC_EXCL    < if file exists, raise, otherwise == inout
        trunc = 2, //H5F_ACC_TRUNC  < if file exists, truncate it and open
        in = 0, //H5F_ACC_RDONLY    < can only read file
        inout = 1  //H5F_ACC_RDWR   < open file for reading and writing
      } mode_t;

      /**
       * Supported types
       */
      typedef enum support_t {
        b = 0, //boolean
        i8, //int8_t
        i16, //int16_t
        i32, //int32_t
        i64, //int64_t
        u8, //uint8_t
        u16, //uint16_t
        u32, //uint32_t
        u64, //uint64_t
        f32, //float
        f64, //double
        f128, //long double
        c64, //std::complex<float>
        c128, //std::complex<double>
        c256, //std::complex<long double>
        s, //std::string
        array, //a marker for when enumeration starts for arrays
        ba, //blitz::Array<bool, N>
        i8a, //blitz::Array<int8_t, N>
        i16a, //blitz::Array<int16_t, N>
        i32a, //blitz::Array<int32_t, N>
        i64a, //blitz::Array<int64_t, N>
        u8a, //blitz::Array<uint8_t, N>
        u16a, //blitz::Array<uint16_t, N>
        u32a, //blitz::Array<uint32_t, N>
        u64a, //blitz::Array<uint64_t, N>
        f32a, //blitz::Array<float, N>
        f64a, //blitz::Array<double, N>
        f128a, //blitz::Array<long double, N>
        c64a, //blitz::Array<std::complex<float>, N>
        c128a, //blitz::Array<std::complex<double>, N>
        c256a //blitz::Array<std::complex<long double>, N>
      } support_t;

      /**
       * Converts a C++ type T into one of the supported types or raise an
       * Unsupported exception.
       */
      template <typename T> support_t supported(const T& value, std::vector<size_t>& shape) {
        //TODO: DONE: raise Unsupported Type
        throw HDF5UnsupportedTypeError();
      }

      /**
       * Condenses information about a certain dataset
       */
      typedef struct typeinfo {

        support_t type; ///< the type of data in the dataset
        std::vector<size_t> shape; ///< the shape of the array. 0 -> scalar

        size_t length; ///< the number of objects in the dataset
        boost::filesystem::path path; ///< full path to this object
        H5::DataSet dataset; ///< the HDF5 Dataset this type points to

        /**
         * Checks if a typeinfo object is compatible with a given object
         */
        template <typename T> bool compatible (const T& value) const {
          typeinfo vinfo(value); //raises for any unsupported type
          return (type == vinfo.type) && (shape == vinfo.shape);
        }

        /**
         * Creates a typeinfo representation from an an existing C++ type
         */
        template <typename T> typeinfo(const T& value) {
          type = supported<T>(value, shape);
        }

      } typeinfo;

    public: //api

      /**
       * Constructor, starts a new HDF5File object giving it a file name and an
       * action: excl/trunc/in/inout
       */
      HDF5File (const boost::filesystem::path& filename, mode_t mode); 

      /**
       * Destructor virtualization
       */
      virtual ~HDF5File();

      /**
       * READING FUNCTIONALITY
       */

      /**
       * Tells if we have a variable with the given name inside the HDF5 file.
       */
      bool contains(const boost::filesystem::path& path);

      /**
       * Describe a certain dataset path
       */
      const typeinfo& describe (const boost::filesystem::path& path);

      /**
       * Returns a scalar with a given name from the file. If the name contains
       * a path like structure, we search for inside the named subgroups.
       */
      template <typename T> T getScalar(const boost::filesystem::path& path,
          size_t index=0) {
        const typeinfo& info = describe(path);
        //TODO: read contents of info.dataset => T for every supported T
        //Note: take "index" into consideration
        T res = 0;
        return res;
      }

      /**
       * Returns an array with a given name from the file. If the name contains
       * a path like structure, we search for inside the named subgroups.
       */
      template <typename T> T getArray(const boost::filesystem::path& path,
          size_t index=0) {
        const typeinfo& info = describe(path); 
        //TODO: read contents of info.dataset => T for every supported T,shape
        //Note: take "index" into consideration
        T res = 0;
        return res;
      }

      /**
       * WRITING FUNCTIONALITY
       */

      /**
       * Inserts a scalar in a certain subdirectory of the file. If that
       * subdirectory does not exist, it is created. If the dataset does not
       * exist, it is created, otherwise, we append to it. In this case, the
       * dimensionality of the scalar has to be compatible with the existing
       * dataset shape (or "dataspace" in HDF5 parlance). You cannot overwrite
       * existing type information. If you want to do this, first unlink and
       * than use one of the add methods.
       */
      template <typename T> void addScalar(const boost::filesystem::path& path,
          const T& value) {
        //TODO: check that T is supported otherwise raise
        //TODO: check that path exists and T is compatible otherwise raise
        //TODO: if path does not exist, create and set dataspace
        //      create a new index entry pointing to the new dataspace
        //TODO: if path exists, add scalar to that path, update index
      }

      /**
       * Inserts a array in a certain subdirectory of the file. If that
       * subdirectory does not exist, it is created. If the dataset does not
       * exist, it is created, otherwise, we append to it. In this case, the
       * dimensionality of the scalar has to be compatible with the existing
       * dataset shape (or "dataspace" in HDF5 parlance). If you want to do
       * this, first unlink and than use one of the add methods.
       */
      template <typename T> void addArray(const boost::filesystem::path& path,
          const T& value) {
        //TODO: check that T is supported otherwise raise
        //TODO: check that path exists and T is compatible otherwise raise
        //TODO: if path does not exist, create and set dataspace
        //      create a new index entry pointing to the new dataspace
        //TODO: if path exists, add array to that path, update index
      }

      /**
       * Unlinks a particular dataset from the file. Please note that this will
       * not erase the data on the current file as that functionality is not
       * provided by HDF5. To actually reclaim the space occupied by the
       * unlinked structure, you must re-save this file to another file. The
       * new file will not contain the data of any dangling datasets (datasets
       * w/o names or links).
       */
      void unlink (const boost::filesystem::path& path);

      /**
       * Copies all reachable objects in another object HDF5File
       */
      void copy (const HDF5File& other);

    private: //not implemented

      /**
       * Copy construct an already opened HDF5File
       */
      HDF5File (const HDF5File& other);

      /**
       * Drop the current settings and load new ones from the other file.
       */
      HDF5File& operator= (const HDF5File& other);

    private: //representation

      boost::filesystem::path m_path; ///< the path to this file
      H5::H5File m_file; ///< the opened HDF5 file
      std::map<boost::filesystem::path, typeinfo> m_index;

  };

/**
  * Specific implementations bind the type T to the support_t enum
  */
#define DEFINE_SUPPORT(T,E) template <> HDF5File::support_t HDF5File::supported<T>(const T& value, std::vector<size_t>& shape) { shape.clear(); return HDF5File::E; }
      DEFINE_SUPPORT(bool,b)
      DEFINE_SUPPORT(int8_t,i8)
      DEFINE_SUPPORT(int16_t,i16)
      DEFINE_SUPPORT(int32_t,i32)
      DEFINE_SUPPORT(int64_t,i64)
      DEFINE_SUPPORT(uint8_t,u8)
      DEFINE_SUPPORT(uint16_t,u16)
      DEFINE_SUPPORT(uint32_t,u32)
      DEFINE_SUPPORT(uint64_t,u64)
      DEFINE_SUPPORT(float,f32)
      DEFINE_SUPPORT(double,f64)
      DEFINE_SUPPORT(long double,f128)
      DEFINE_SUPPORT(std::complex<float>,c64)
      DEFINE_SUPPORT(std::complex<double>,c128)
      DEFINE_SUPPORT(std::complex<long double>,c256)
      DEFINE_SUPPORT(std::string,s)
#undef DEFINE_SUPPORT

#define DEFINE_BZ_SUPPORT_N(T,E,N) template <> HDF5File::support_t \
      HDF5File::supported<blitz::Array<T,N> >(const blitz::Array<T,N>& value, std::vector<size_t>& shape) { \
        if (N > Torch::core::array::N_MAX_DIMENSIONS_ARRAY) throw HDF5UnsupportedDimensionError(N); \
        shape.clear(); \
        for (int i=0; i<N; ++i) shape.push_back(value.extent(i)); \
        return HDF5File::E; \
      }

#define DEFINE_BZ_SUPPORT(T,E) \
      DEFINE_BZ_SUPPORT_N(T,E,1) \
      DEFINE_BZ_SUPPORT_N(T,E,2) \
      DEFINE_BZ_SUPPORT_N(T,E,3) \
      DEFINE_BZ_SUPPORT_N(T,E,4) 

      DEFINE_BZ_SUPPORT(bool,ba)
      DEFINE_BZ_SUPPORT(int8_t,i8a)
      DEFINE_BZ_SUPPORT(int16_t,i16a)
      DEFINE_BZ_SUPPORT(int32_t,i32a)
      DEFINE_BZ_SUPPORT(int64_t,i64a)
      DEFINE_BZ_SUPPORT(uint8_t,u8a)
      DEFINE_BZ_SUPPORT(uint16_t,u16a)
      DEFINE_BZ_SUPPORT(uint32_t,u32a)
      DEFINE_BZ_SUPPORT(uint64_t,u64a)
      DEFINE_BZ_SUPPORT(float,f32a)
      DEFINE_BZ_SUPPORT(double,f64a)
      DEFINE_BZ_SUPPORT(long double,f128a)
      DEFINE_BZ_SUPPORT(std::complex<float>,c64a)
      DEFINE_BZ_SUPPORT(std::complex<double>,c128a)
      DEFINE_BZ_SUPPORT(std::complex<long double>,c256a)
#undef DEFINE_BZ_SUPPORT
#undef DEFINE_BZ_SUPPORT_N

}}

#endif /* TORCH_DATABASE_HDF5FILE_H */
