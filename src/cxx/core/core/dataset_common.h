/**
 * @file src/cxx/core/core/dataset_common.h
 * @author <a href="mailto:Laurent.El-Shafey@idiap.ch">Laurent El Shafey</a> 
 *
 * @brief This file contains standard definitions for the Dataset 
 * implementation
 *
 */

#ifndef TORCH_CORE_DATASET_COMMON_H
#define TORCH_CORE_DATASET_COMMON_H

#include "core/Exception.h"

namespace Torch {
/**
 * \ingroup libcore_api
 * @{
 *
 */
  namespace core {

    /**
     * string for the XML attributes
     */
    namespace db {
      extern const char *dataset;
      extern const char *arrayset;
      extern const char *external_arrayset;
      extern const char *relationset;
      extern const char *version;
      extern const char *id;
      extern const char *role;
      extern const char *elementtype;
      extern const char *shape;
      extern const char *loader;
      extern const char *file;
      extern const char *array;
      extern const char *external_array;
      extern const char *name;
      extern const char *rule;
      extern const char *relation;
      extern const char *member;
      extern const char *arrayset_member;
      extern const char *arrayset_role;
      extern const char *min;
      extern const char *max;
      extern const char *array_id;
      extern const char *arrayset_id;

      // elementtype
      extern const char *t_bool;
      extern const char *t_int8;
      extern const char *t_int16;
      extern const char *t_int32;
      extern const char *t_int64;
      extern const char *t_uint8;
      extern const char *t_uint16;
      extern const char *t_uint32;
      extern const char *t_uint64;
      extern const char *t_float32;
      extern const char *t_float64;
      extern const char *t_float128;
      extern const char *t_complex64;
      extern const char *t_complex128;
      extern const char *t_complex256;

      // loader
      extern const char *l_blitz;
      extern const char *l_tensor;
      extern const char *l_bindata;
      extern const char *l_byextension;
    }

    typedef enum LoaderType { l_unknown, l_blitz, l_tensor, l_bindata }
      LoaderType;

    class IndexError: public Exception { };
    class NDimensionError: public Exception { };
    class TypeError: public Exception { };
    class NonExistingElement: public Exception { };

  }
/**
 * @}
 */
}

#endif /* TORCH_CORE_DATASET_COMMON_H */

