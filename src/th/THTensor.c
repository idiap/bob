#include "THTensor.h"
#include "THTensorApply.h"

#define TYPE char
#define CAP_TYPE Char
#include "THTensorGen.c"
#undef TYPE
#undef CAP_TYPE

#define TYPE short
#define CAP_TYPE Short
#include "THTensorGen.c"
#undef TYPE
#undef CAP_TYPE

#define TYPE int
#define CAP_TYPE Int
#include "THTensorGen.c"
#undef TYPE
#undef CAP_TYPE

#define TYPE long
#define CAP_TYPE Long
#include "THTensorGen.c"
#undef TYPE
#undef CAP_TYPE

#define TYPE float
#define CAP_TYPE Float
#include "THTensorGen.c"
#undef TYPE
#undef CAP_TYPE

#define TYPE double
#define CAP_TYPE Double
#define DEFAULT_TENSOR
#include "THTensorGen.c"
#undef TYPE
#undef CAP_TYPE
#undef DEFAULT_TENSOR
