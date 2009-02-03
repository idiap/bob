#include "THStorage.h"

/* Stuff for mapped files */
#ifdef _WIN32
#include <windows.h>
#endif

#if HAVE_MMAP
#include <sys/types.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#endif
/* End of stuff for mapped files */

#define TYPE char
#define CAP_TYPE Char
#include "THStorageGen.c"
#undef TYPE
#undef CAP_TYPE

#define TYPE short
#define CAP_TYPE Short
#include "THStorageGen.c"
#undef TYPE
#undef CAP_TYPE

#define TYPE int
#define CAP_TYPE Int
#include "THStorageGen.c"
#undef TYPE
#undef CAP_TYPE

#define TYPE long
#define CAP_TYPE Long
#include "THStorageGen.c"
#undef TYPE
#undef CAP_TYPE

#define TYPE float
#define CAP_TYPE Float
#include "THStorageGen.c"
#undef TYPE
#undef CAP_TYPE

#define TYPE double
#define CAP_TYPE Double
#include "THStorageGen.c"
#undef TYPE
#undef CAP_TYPE
