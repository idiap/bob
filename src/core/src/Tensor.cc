#include "core/Tensor.h"

namespace Torch {

const char *str_TensorTypeName[] = {"char", "short", "int", "long", "float", "double"};

#if TORCH5SPRO_TENSOR_TYPE == 1
// Get the number of elements (over all dimensions)
long Tensor::sizeAll() const
{
	long count = 1;
	for (int i = 0; i < nDimension(); i ++)
		count *= size(i);
	return count;
}

#define DATATYPE 0
#define TYPE char
#define CAP_TYPE Char
#define TYPE_FORMAT "%d"
#include "TensorGen.cc"
#undef TYPE_FORMAT
#undef TYPE
#undef CAP_TYPE
#undef DATATYPE

#define DATATYPE 1
#define TYPE short
#define CAP_TYPE Short
#define TYPE_FORMAT "%d"
#include "TensorGen.cc"
#undef TYPE_FORMAT
#undef TYPE
#undef CAP_TYPE
#undef DATATYPE

#define DATATYPE 2
#define TYPE int
#define CAP_TYPE Int
#define TYPE_FORMAT "%d"
#include "TensorGen.cc"
#undef TYPE_FORMAT
#undef TYPE
#undef CAP_TYPE
#undef DATATYPE

#define DATATYPE 3
#define TYPE long
#define CAP_TYPE Long
#define TYPE_FORMAT "%ld"
#include "TensorGen.cc"
#undef TYPE_FORMAT
#undef TYPE
#undef CAP_TYPE
#undef DATATYPE

#define DATATYPE 4
#define TYPE float
#define CAP_TYPE Float
#define TYPE_FORMAT "%f"
#include "TensorGen.cc"
#undef TYPE_FORMAT
#undef TYPE
#undef CAP_TYPE
#undef DATATYPE

#define DATATYPE 5
#define TYPE double
#define CAP_TYPE Double
#define TYPE_FORMAT "%g"
#define DEFAULT_TENSOR
#include "TensorGen.cc"
#undef TYPE_FORMAT
#undef TYPE
#undef CAP_TYPE
#undef DEFAULT_TENSOR
#undef DATATYPE

#endif

}

#if TORCH5SPRO_TENSOR_TYPE > 1 
#include "TensorWrapper.cc"
#endif
