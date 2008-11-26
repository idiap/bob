#ifndef TH_STORAGE_INC
#define TH_STORAGE_INC

#include "THGeneral.h"

#define TYPE char
#define CAP_TYPE Char
#include "THStorageGen.h"
#undef TYPE
#undef CAP_TYPE

#define TYPE short
#define CAP_TYPE Short
#include "THStorageGen.h"
#undef TYPE
#undef CAP_TYPE

#define TYPE int
#define CAP_TYPE Int
#include "THStorageGen.h"
#undef TYPE
#undef CAP_TYPE

#define TYPE long
#define CAP_TYPE Long
#include "THStorageGen.h"
#undef TYPE
#undef CAP_TYPE

#define TYPE float
#define CAP_TYPE Float
#include "THStorageGen.h"
#undef TYPE
#undef CAP_TYPE

#define TYPE double
#define CAP_TYPE Double
#include "THStorageGen.h"
#undef TYPE
#undef CAP_TYPE

#endif
