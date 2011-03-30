/**
 * @author <a href="mailto:andre.anjos@idiap.ch">Andre Anjos</a> 
 * @date Wed 30 Mar 21:06:28 2011 
 *
 * @brief Torch support for HDF5 files. HDF5 is a open standard for
 * self-describing data files. You can get more information in this webpage:
 * http://www.hdfgroup.org/HDF5
 */

#ifndef TORCH_DATABASE_HDF5FILE_H 
#define TORCH_DATABASE_HDF5FILE_H

#include <typeinfo>
#include <map>
#include <boost/filesystem.hpp>
#include <boost/make_shared.hpp>
#include <blitz/array.h>
#include <blitz/tinyvec-et.h>
#include <H5Cpp.h>

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
        excl = H5F_ACC_EXCL
          trunc = H5F_ACC_TRUNC,
        in = H5F_ACC_RDONLY,
        inout = H5F_ACC_RDWR,
      } mode_t;

      /**
       * Supported types
       */
      typedef enum scalar_t {
        b = 0, //boolean
        i8 = 1, //int8_t
        i16 = 2, //int16_t
        i32 = 3, //int32_t
        i64 = 4, //int64_t
        u8 = 5, //uint8_t
        u16 = 6, //uint16_t
        u32 = 7, //uint32_t
        u64 = 8, //uint64_t
        f32 = 9, //float
        f64 = 10, //double
        f128 = 11, //long double
        c64 = 12, //std::complex<float>
        c128 = 13, //std::complex<double>
        c256 = 14, //std::complex<long double>
        ch = 15, //char
        s = 16 //std::string
      } scalar_t;

      /**
       * Condenses information about a certain dataset
       */
      struct typeinfo {

        boost::filesystem::path path; ///< full path to this object
        scalar_t scalar; ///< the type of scalar in the dataset
        size_t rank; ///< the number of dimensions: 0 means is a scalar
        std::vector<size_t> shape; ///< the shape of the array if rank != 0
        size_t length; ///< the number of objects in the dataset
        const std::type_info& cxxinfo; ///< equivalent C++ runtime type

        /**
         * Checks if a typeinfo object is compatible with a given object
         */
        template <typename T> bool compatible (const T& value) const {
          if (cxxinfo != typeid(value)) return false;
          if (rank) { //is array: verify shape is compatible
            if (shape.size() != value.shape().length()) return false;
            for (size_t i=0; i<shape.size(); ++i) {
              if (shape[i] != value.shape()[i]) return false;
            }
          }
          return true;
        }

      };

      /**
       * Constructor, starts a new HDF5File object giving it a file name and an
       * action: excl/trunc/in/inout
       */
      HDF5File (const boost::filesystem::path& filename, mode_t mode); 

      /**
       * Copy construct an already opened HDF5File
       */
      HDF5File (const HDF5File& other);

      /**
       * Destructor virtualization
       */
      virtual ~HDF5File();

      /**
       * Drop the current settings and load new ones from the other file.
       */
      HDF5File& operator= (const HDF5File& other);

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
      boost::shared_ptr<typeinfo> describe (const boost::filesystem::path& path);

      /**
       * Returns a scalar with a given name from the file. If the name contains
       * a path like structure, we search for inside the named subgroups.
       */
      template <typename T> T getScalar(const boost::filesystem::path& path,
          size_t index=0);

      /**
       * Returns an array with a given name from the file. If the name contains
       * a path like structure, we search for inside the named subgroups.
       */
      template <typename T> T getArray(const boost::filesystem::path& path,
          size_t index=0);

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
          const T& value);

      /**
       * Inserts a array in a certain subdirectory of the file. If that
       * subdirectory does not exist, it is created. If the dataset does not
       * exist, it is created, otherwise, we append to it. In this case, the
       * dimensionality of the scalar has to be compatible with the existing
       * dataset shape (or "dataspace" in HDF5 parlance). If you want to do
       * this, first unlink and than use one of the add methods.
       */
      template <typename T> void addArray(const boost::filesystem::path& path,
          const T& value);

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

    private: //representation

      boost::filesystem::path m_path;
      //HDF5File??
      std::map<boost::filesystem::path, typeinfo> m_index;

  };

}}

#endif /* TORCH_DATABASE_HDF5FILE_H */
