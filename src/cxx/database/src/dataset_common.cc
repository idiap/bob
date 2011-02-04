/**
 * @file src/cxx/core/src/dataset_common.cc
 * @author <a href="mailto:Laurent.El-Shafey@idiap.ch">Laurent El Shafey</a> 
 *
 * @brief This file defines const string for the Dataset implementation
 *
 */

#include "core/dataset_common.h"

namespace Torch {
  namespace core {

    /**
     * string for the XML attributes
     */
    namespace db {
      const char *dataset           = "dataset";
      const char *arrayset          = "arrayset";
      const char *external_arrayset = "external-arrayset";
      const char *relationset       = "relationset";
      const char *version           = "version";
      const char *id                = "id";
      const char *role              = "role";
      const char *elementtype       = "elementtype";
      const char *shape             = "shape";
      const char *codec             = "codec";
      const char *file              = "file";
      const char *array             = "array";
      const char *external_array    = "external-array";
      const char *name              = "name";
      const char *rule              = "rule";
      const char *relation          = "relation";
      const char *member            = "member";
      const char *arrayset_member   = "arrayset-member";
      const char *arrayset_role     = "arrayset-role";
      const char *min               = "min";
      const char *max               = "max";
      const char *array_id          = "array-id";
      const char *arrayset_id       = "arrayset-id";

      // elementtype
      const char *t_bool        = "bool";
      const char *t_int8        = "int8";
      const char *t_int16       = "int16";
      const char *t_int32       = "int32";
      const char *t_int64       = "int64";
      const char *t_uint8       = "uint8";
      const char *t_uint16      = "uint16";
      const char *t_uint32      = "uint32";
      const char *t_uint64      = "uint64";
      const char *t_float32     = "float32";
      const char *t_float64     = "float64";
      const char *t_float128    = "float128";
      const char *t_complex64   = "complex64";
      const char *t_complex128  = "complex128";
      const char *t_complex256  = "complex256";

      // codec
      const char *c_blitz       = "blitz";
      const char *c_tensor      = "tensor";
      const char *c_bindata     = "bindata";
      const char *c_byextension = "byextension";
    }

  }
}

