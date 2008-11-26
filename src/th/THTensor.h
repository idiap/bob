#ifndef TH_TENSOR_INC
#define TH_TENSOR_INC

#include "THStorage.h"

#define TYPE char
#define CAP_TYPE Char
#include "THTensorGen.h"
#undef TYPE
#undef CAP_TYPE

#define TYPE short
#define CAP_TYPE Short
#include "THTensorGen.h"
#undef TYPE
#undef CAP_TYPE

#define TYPE int
#define CAP_TYPE Int
#include "THTensorGen.h"
#undef TYPE
#undef CAP_TYPE

#define TYPE long
#define CAP_TYPE Long
#include "THTensorGen.h"
#undef TYPE
#undef CAP_TYPE

#define TYPE float
#define CAP_TYPE Float
#include "THTensorGen.h"
#undef TYPE
#undef CAP_TYPE

#define TYPE double
#define CAP_TYPE Double
#define DEFAULT_TENSOR
#include "THTensorGen.h"
#undef TYPE
#undef CAP_TYPE
#undef DEFAULT_TENSOR

#endif
