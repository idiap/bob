/* Storage type */
#define STORAGE_T_(TYPE) TH##TYPE##Storage
#define STORAGE_T(TYPE) STORAGE_T_(TYPE)
#define STORAGE STORAGE_T(CAP_TYPE)

/* Tensor type */
#define TENSOR_T_(TYPE) TH##TYPE##Tensor
#define TENSOR_T(TYPE) TENSOR_T_(TYPE)
#define TENSOR TENSOR_T(CAP_TYPE)

/* Function name for a Storage */
#define STORAGE_FUNC_TN_(TYPE,NAME) TH##TYPE##Storage_##NAME
#define STORAGE_FUNC_TN(TYPE, NAME) STORAGE_FUNC_TN_(TYPE,NAME) 
#define STORAGE_FUNC(NAME) STORAGE_FUNC_TN(CAP_TYPE, NAME)

/* Function name for a Tensor */
#define TENSOR_FUNC_TN_(TYPE,NAME) TH##TYPE##Tensor_##NAME
#define TENSOR_FUNC_TN(TYPE, NAME) TENSOR_FUNC_TN_(TYPE,NAME) 
#define TENSOR_FUNC(NAME) TENSOR_FUNC_TN(CAP_TYPE, NAME)

/* For the default Tensor type, we simplify the naming */
#ifdef DEFAULT_TENSOR
#undef TENSOR
#undef TENSOR_FUNC
#define TENSOR THTensor
#define TENSOR_FUNC(NAME) TENSOR_FUNC_TN(, NAME)
#endif

static void TENSOR_FUNC(reinit)(TENSOR *tensor, STORAGE *storage, long storageOffset, int nDimension, long *size, long *stride);

/* Empty init */
TENSOR *TENSOR_FUNC(new)(void)
{
  TENSOR *tensor = THAlloc(sizeof(TENSOR));
  tensor->size = NULL;
  tensor->stride = NULL;
  tensor->nDimension = 0;
  tensor->storage = NULL;
  tensor->storageOffset = 0;
  tensor->ownStorage = 0;
  tensor->refcount = 1;
  return tensor;
}

/* Pointer-copy init */
TENSOR *TENSOR_FUNC(newWithTensor)(TENSOR *src)
{
  TENSOR *tensor = TENSOR_FUNC(new)();
  TENSOR_FUNC(reinit)(tensor, src->storage, src->storageOffset, src->nDimension, src->size, src->stride);
  return tensor;
}

/* Storage init */
TENSOR *TENSOR_FUNC(newWithStorage)(STORAGE *storage, long storageOffset, int nDimension, long *size, long *stride)
{
  TENSOR *tensor = TENSOR_FUNC(new)();
  TENSOR_FUNC(reinit)(tensor, storage, storageOffset, nDimension, size, stride);
  return tensor;
}

TENSOR *TENSOR_FUNC(newWithStorage1d)(STORAGE *storage, long storageOffset,
                               long size0, long stride0)
{
  return TENSOR_FUNC(newWithStorage4d)(storage, storageOffset, size0, stride0, -1, -1,  -1, -1,  -1, -1);
}

TENSOR *TENSOR_FUNC(newWithStorage2d)(STORAGE *storage, long storageOffset,
                               long size0, long stride0,
                               long size1, long stride1)
{
  return TENSOR_FUNC(newWithStorage4d)(storage, storageOffset, size0, stride0, size1, stride1,  -1, -1,  -1, -1);
}

TENSOR *TENSOR_FUNC(newWithStorage3d)(STORAGE *storage, long storageOffset,
                               long size0, long stride0,
                               long size1, long stride1,
                               long size2, long stride2)
{
  return TENSOR_FUNC(newWithStorage4d)(storage, storageOffset, size0, stride0, size1, stride1,  size2, stride2,  -1, -1);
}

TENSOR *TENSOR_FUNC(newWithStorage4d)(STORAGE *storage, long storageOffset,
                               long size0, long stride0,
                               long size1, long stride1,
                               long size2, long stride2,
                               long size3, long stride3)
{
  long size[4];
  long stride[4];
  size[0] = size0;
  size[1] = size1;
  size[2] = size2;
  size[3] = size3;
  stride[0] = stride0;
  stride[1] = stride1;
  stride[2] = stride2;
  stride[3] = stride3;  
  return TENSOR_FUNC(newWithStorage)(storage, storageOffset, 4, size, stride);
}

/* Normal init */
TENSOR *TENSOR_FUNC(newWithSize)(int nDimension, long *size, long *stride)
{
  return TENSOR_FUNC(newWithStorage)(NULL, 0, nDimension, size, stride);
}

TENSOR *TENSOR_FUNC(newWithSize1d)(long size0)
{
  return TENSOR_FUNC(newWithSize4d)(size0, -1, -1, -1);
}

TENSOR *TENSOR_FUNC(newWithSize2d)(long size0, long size1)
{
  return TENSOR_FUNC(newWithSize4d)(size0, size1, -1, -1);
}

TENSOR *TENSOR_FUNC(newWithSize3d)(long size0, long size1, long size2)
{
  return TENSOR_FUNC(newWithSize4d)(size0, size1, size2, -1);
}

TENSOR *TENSOR_FUNC(newWithSize4d)(long size0, long size1, long size2, long size3)
{
  long size[4];
  size[0] = size0;
  size[1] = size1;
  size[2] = size2;
  size[3] = size3;
  return TENSOR_FUNC(newWithSize)(4, size, NULL);
}

/* Set */
void TENSOR_FUNC(setTensor)(TENSOR *tensor, TENSOR *src)
{
  TENSOR_FUNC(reinit)(tensor, src->storage, src->storageOffset, src->nDimension, src->size, src->stride);
}

void TENSOR_FUNC(setStorage)(TENSOR *tensor, STORAGE *storage, long storageOffset, int nDimension, long *size, long *stride)
{
  TENSOR_FUNC(reinit)(tensor, storage, storageOffset, nDimension, size, stride);
}

void TENSOR_FUNC(setStorage1d)(TENSOR *tensor, STORAGE *storage, long storageOffset,
                        long size0, long stride0)
{
  TENSOR_FUNC(setStorage4d)(tensor, storage, storageOffset, size0, stride0, -1, -1, -1, -1, -1, -1);
}

void TENSOR_FUNC(setStorage2d)(TENSOR *tensor, STORAGE *storage, long storageOffset,
                        long size0, long stride0,
                        long size1, long stride1)
{
  TENSOR_FUNC(setStorage4d)(tensor, storage, storageOffset, size0, stride0, size1, stride1, -1, -1, -1, -1);
}

void TENSOR_FUNC(setStorage3d)(TENSOR *tensor, STORAGE *storage, long storageOffset,
                        long size0, long stride0,
                        long size1, long stride1,
                        long size2, long stride2)
{
  TENSOR_FUNC(setStorage4d)(tensor, storage, storageOffset, size0, stride0, size1, stride1, size2, stride2, -1, -1);
}

void TENSOR_FUNC(setStorage4d)(TENSOR *tensor, STORAGE *storage, long storageOffset,
                        long size0, long stride0,
                        long size1, long stride1,
                        long size2, long stride2,
                        long size3, long stride3)
{
  long size[4];
  long stride[4];
  size[0] = size0;
  size[1] = size1;
  size[2] = size2;
  size[3] = size3;
  stride[0] = stride0;
  stride[1] = stride1;
  stride[2] = stride2;
  stride[3] = stride3;
  TENSOR_FUNC(setStorage)(tensor, storage, storageOffset, 4, size, stride);
}

void TENSOR_FUNC(set1d)(TENSOR *tensor, long x0, TYPE value)
{
  THArgCheck(tensor->nDimension == 1, 1, "tensor must have one dimension");
  THArgCheck( (x0 >= 0) && (x0 < tensor->size[0]), 2, "out of range");
  (tensor->storage->data+tensor->storageOffset)[x0*tensor->stride[0]] = value;
}

TYPE TENSOR_FUNC(get1d)(TENSOR *tensor, long x0)
{
  THArgCheck(tensor->nDimension == 1, 1, "tensor must have one dimension");
  THArgCheck( (x0 >= 0) && (x0 < tensor->size[0]), 2, "out of range");
  return (tensor->storage->data+tensor->storageOffset)[x0*tensor->stride[0]];
}

void TENSOR_FUNC(set2d)(TENSOR *tensor, long x0, long x1, TYPE value)
{
  THArgCheck(tensor->nDimension == 2, 1, "tensor must have two dimensions");
  THArgCheck((x0 >= 0) && (x0 < tensor->size[0]) && (x1 >= 0) && (x1 < tensor->size[1]), 2, "out of range");
  (tensor->storage->data+tensor->storageOffset)[x0*tensor->stride[0]+x1*tensor->stride[1]] = value;
}

TYPE TENSOR_FUNC(get2d)(TENSOR *tensor, long x0, long x1)
{
  THArgCheck(tensor->nDimension == 2, 1, "tensor must have two dimensions");
  THArgCheck((x0 >= 0) && (x0 < tensor->size[0]) && (x1 >= 0) && (x1 < tensor->size[1]), 2, "out of range");
  return( (tensor->storage->data+tensor->storageOffset)[x0*tensor->stride[0]+x1*tensor->stride[1]] );
}

void TENSOR_FUNC(set3d)(TENSOR *tensor, long x0, long x1, long x2, TYPE value)
{
  THArgCheck(tensor->nDimension == 3, 1, "tensor must have three dimensions");
  THArgCheck( (x0 >= 0) && (x0 < tensor->size[0]) && (x1 >= 0) && (x1 < tensor->size[1]) && (x2 >= 0) && (x2 < tensor->size[2]), 2, "out of range");
  (tensor->storage->data+tensor->storageOffset)[x0*tensor->stride[0]+x1*tensor->stride[1]+x2*tensor->stride[2]] = value;
}

TYPE TENSOR_FUNC(get3d)(TENSOR *tensor, long x0, long x1, long x2)
{
  THArgCheck(tensor->nDimension == 3, 1, "tensor must have three dimensions");
  THArgCheck( (x0 >= 0) && (x0 < tensor->size[0]) && (x1 >= 0) && (x1 < tensor->size[1]) && (x2 >= 0) && (x2 < tensor->size[2]), 2, "out of range");
  return( (tensor->storage->data+tensor->storageOffset)[x0*tensor->stride[0]+x1*tensor->stride[1]+x2*tensor->stride[2]] );
}

void TENSOR_FUNC(set4d)(TENSOR *tensor, long x0, long x1, long x2, long x3, TYPE value)
{
  THArgCheck(tensor->nDimension == 4, 1, "tensor must have four dimensions");
  THArgCheck((x0 >= 0) && (x0 < tensor->size[0]) && (x1 >= 0) && (x1 < tensor->size[1]) && (x2 >= 0) && (x2 < tensor->size[2]) && (x3 >= 0) && (x3 < tensor->size[3]), 2, "out of range");
  (tensor->storage->data+tensor->storageOffset)[x0*tensor->stride[0]+x1*tensor->stride[1]+x2*tensor->stride[2]+x3*tensor->stride[3]] = value;
}

TYPE TENSOR_FUNC(get4d)(TENSOR *tensor, long x0, long x1, long x2, long x3)
{
  THArgCheck(tensor->nDimension == 4, 1, "tensor must have four dimensions");
  THArgCheck((x0 >= 0) && (x0 < tensor->size[0]) && (x1 >= 0) && (x1 < tensor->size[1]) && (x2 >= 0) && (x2 < tensor->size[2]) && (x3 >= 0) && (x3 < tensor->size[3]), 2, "out of range");
  return( (tensor->storage->data+tensor->storageOffset)[x0*tensor->stride[0]+x1*tensor->stride[1]+x2*tensor->stride[2]+x3*tensor->stride[3]] );
}

/* Resize */
void TENSOR_FUNC(resizeAs)(TENSOR *tensor, TENSOR *src)
{
  int isSame = 0;
  int d;
  if(tensor->nDimension == src->nDimension)
  {
    isSame = 1;
    for(d = 0; d < tensor->nDimension; d++)
    {
      if(tensor->size[d] != src->size[d])
      {
        isSame = 0;
        break;
      }
    }
  }
  if(!isSame)
    TENSOR_FUNC(reinit)(tensor, NULL, 0, src->nDimension, src->size, NULL);
}

void TENSOR_FUNC(resize)(TENSOR *tensor, int nDimension, long *size)
{
  int isSame = 0;
  if(nDimension == tensor->nDimension)
  {
    int d;
    isSame = 1;
    for(d = 0; d < tensor->nDimension; d++)
    {
      if(tensor->size[d] != size[d])
      {
        isSame = 0;
        break;
      }
    }
  }
  if(!isSame)
    TENSOR_FUNC(reinit)(tensor, NULL, 0, nDimension, size, NULL);
}

void TENSOR_FUNC(resize1d)(TENSOR *tensor, long size0)
{
  TENSOR_FUNC(resize4d)(tensor, size0, -1, -1, -1);
}

void TENSOR_FUNC(resize2d)(TENSOR *tensor, long size0, long size1)
{
  TENSOR_FUNC(resize4d)(tensor, size0, size1, -1, -1);
}

void TENSOR_FUNC(resize3d)(TENSOR *tensor, long size0, long size1, long size2)
{
  TENSOR_FUNC(resize4d)(tensor, size0, size1, size2, -1);
}

void TENSOR_FUNC(resize4d)(TENSOR *tensor, long size0, long size1, long size2, long size3)
{
  long size[4];
  size[0] = size0;
  size[1] = size1;
  size[2] = size2;
  size[3] = size3;
  TENSOR_FUNC(resize)(tensor, 4, size);
}

void TENSOR_FUNC(narrow)(TENSOR *tensor, TENSOR *src, int dimension, long firstIndex, long size)
{
  if(!src)
    src = tensor;

  THArgCheck( (dimension >= 0) && (dimension < src->nDimension), 2, "out of range");
  THArgCheck( (firstIndex >= 0) && (firstIndex < src->size[dimension]), 3, "out of range");
  THArgCheck( (size > 0) && (firstIndex+size <= src->size[dimension]), 4, "out of range");

  TENSOR_FUNC(setTensor)(tensor, src);
  
  if(firstIndex > 0)
    tensor->storageOffset += firstIndex*tensor->stride[dimension];

  tensor->size[dimension] = size;
}


void TENSOR_FUNC(select)(TENSOR *tensor, TENSOR *src, int dimension, long sliceIndex)
{
  int d;

  if(!src)
    src = tensor;

  THArgCheck(src->nDimension > 1, 1, "cannot select on a vector");
  THArgCheck((dimension >= 0) && (dimension < src->nDimension), 2, "out of range");
  THArgCheck((sliceIndex >= 0) && (sliceIndex < src->size[dimension]), 3, "out of range");

  TENSOR_FUNC(narrow)(tensor, src, dimension, sliceIndex, 1);
  for(d = dimension; d < tensor->nDimension-1; d++)
  {
    tensor->size[d] = src->size[d+1];
    tensor->stride[d] = src->stride[d+1];
  }
  tensor->nDimension--;
}


void TENSOR_FUNC(transpose)(TENSOR *tensor, TENSOR *src, int dimension1, int dimension2)
{
  long z;

  if(!src) 
    src = tensor;

  THArgCheck( (dimension1 >= 0) && (dimension1 < src->nDimension), 2, "out of range");
  THArgCheck( (dimension2 >= 0) && (dimension2 < src->nDimension), 3, "out of range");

  TENSOR_FUNC(setTensor)(tensor, src);

  if(dimension1 == dimension2)
	  return;
 
  z = tensor->stride[dimension1];
  tensor->stride[dimension1] = tensor->stride[dimension2];
  tensor->stride[dimension2] = z;
  z = tensor->size[dimension1];
  tensor->size[dimension1] = tensor->size[dimension2];
  tensor->size[dimension2] = z;
}

void TENSOR_FUNC(unfold)(TENSOR *tensor, TENSOR *src, int dimension, long size, long step)
{
  long *newSize;
  long *newStride;
  int d;

  if(!src)
    src = tensor;

  THArgCheck( (src->nDimension > 0), 1, "cannot unfold an empty tensor");
  THArgCheck(dimension < src->nDimension, 2, "out of range");
  THArgCheck(size <= src->size[dimension], 3, "out of range");
  THArgCheck(step > 0, 4, "invalid step");

  TENSOR_FUNC(setTensor)(tensor, src);

  newSize = THAlloc(sizeof(long)*(tensor->nDimension+1));
  newStride = THAlloc(sizeof(long)*(tensor->nDimension+1));

  newSize[tensor->nDimension] = size;
  newStride[tensor->nDimension] = tensor->stride[dimension];
  for(d = 0; d < tensor->nDimension; d++)
  {
    if(d == dimension)
    {
      newSize[d] = (tensor->size[d] - size) / step + 1;
      newStride[d] = step*tensor->stride[d];
    }
    else
    {
      newSize[d] = tensor->size[d];
      newStride[d] = tensor->stride[d];
    }
  }

  TENSOR_FUNC(reinit)(tensor, tensor->storage, tensor->storageOffset, tensor->nDimension+1, newSize, newStride);
  THFree(newSize);
  THFree(newStride);
}

/* fill */
void TENSOR_FUNC(fill)(TENSOR *tensor, TYPE value)
{
  TH_TENSOR_APPLY(TYPE, tensor, *tensor_p = value;);
}

/* is contiguous? [a bit like in TnXIterator] */
int TENSOR_FUNC(isContiguous)(TENSOR *tensor)
{
  long z = 1;
  int d;
  for(d = 0; d < tensor->nDimension; d++)
  {
    if(tensor->stride[d] == z)
      z *= tensor->size[d];
    else
      return 0;
  }
  return 1;
}

long TENSOR_FUNC(nElement)(TENSOR *tensor)
{
  if(tensor->nDimension == 0)
    return 0;
  else
  {
    long nElement = 1;
    int d;
    for(d = 0; d < tensor->nDimension; d++)
      nElement *= tensor->size[d];
    return nElement;
  }
}

void TENSOR_FUNC(retain)(TENSOR *tensor)
{
  if(tensor)
    ++tensor->refcount;
}

void TENSOR_FUNC(free)(TENSOR *tensor)
{
  if(!tensor)
    return;

  if(--tensor->refcount == 0)
  {
    THFree(tensor->size);
    THFree(tensor->stride);
    STORAGE_FUNC(free)(tensor->storage);
    THFree(tensor);
  }
}

/*******************************************************************************/

/* This one does everything except coffee */
static void TENSOR_FUNC(reinit)(TENSOR *tensor, STORAGE *storage, long storageOffset, int nDimension, long *size, long *stride)
{
  int d;
  int nDimension_;
  long totalSize;

  /* Storage stuff */
  if(storageOffset < 0)
    THError("Tensor: invalid storage offset");

  tensor->storageOffset = storageOffset;

  if(storage) /* new storage ? */
  {
    if(storage != tensor->storage) /* really new?? */
    {
      if(tensor->storage)
        STORAGE_FUNC(free)(tensor->storage);
      
      tensor->storage = storage;
      STORAGE_FUNC(retain)(tensor->storage);
      tensor->ownStorage = 0;
    } /* else we had already this storage, so we keep it */
  }
  else
  {
    if(tensor->storage)
    {
      if(!tensor->ownStorage)
      {
        STORAGE_FUNC(free)(tensor->storage);
        tensor->storage = STORAGE_FUNC(new)();
        tensor->ownStorage = 1;
      } /* else we keep our storage */
    }
    else
    {
      tensor->storage = STORAGE_FUNC(new)();
      tensor->ownStorage = 1;
    }
  }

  /* nDimension, size and stride */
  nDimension_ = 0;
  for(d = 0; d < nDimension; d++)
  {
    if(size[d] > 0)
      nDimension_++;
    else
      break;
  }
  nDimension = nDimension_;

  if(nDimension > 0)
  {
    if(nDimension > tensor->nDimension)
    {
      THFree(tensor->size);
      THFree(tensor->stride);
      tensor->size = THAlloc(sizeof(long)*nDimension);
      tensor->stride = THAlloc(sizeof(long)*nDimension);
    }
    tensor->nDimension = nDimension;

    totalSize = 1;
    for(d = 0; d < tensor->nDimension; d++)
    {
      tensor->size[d] = size[d];
      if(stride && (stride[d] >= 0) )
        tensor->stride[d] = stride[d];
      else
      {
        if(d == 0)
          tensor->stride[d] = 1;
        else
          tensor->stride[d] = tensor->size[d-1]*tensor->stride[d-1];
      }
      totalSize += (tensor->size[d]-1)*tensor->stride[d];
    }
    
    if(totalSize+storageOffset > tensor->storage->size) /* if !ownStorage, that might be a problem! */
    {
      if(!tensor->ownStorage)
        THError("Tensor: trying to resize a storage which is not mine");
      STORAGE_FUNC(resize)(tensor->storage, totalSize+storageOffset, 0);
    }
  }
  else
  {
    tensor->nDimension = 0;    
  }
}

inline TYPE* TENSOR_FUNC(dataPtr)(TENSOR *tensor)
{
  return tensor->storage->data+tensor->storageOffset;
}
    
inline TYPE* TENSOR_FUNC(dataPtr1d)(TENSOR *tensor, long i0)
{
  return tensor->storage->data+tensor->storageOffset+i0*tensor->stride[0];
}

inline TYPE* TENSOR_FUNC(dataPtr2d)(TENSOR *tensor, long i0, long i1)
{
  return tensor->storage->data+tensor->storageOffset+i0*tensor->stride[0]+i1*tensor->stride[1];
}

inline TYPE* TENSOR_FUNC(dataPtr3d)(TENSOR *tensor, long i0, long i1, long i2)
{
  return tensor->storage->data+tensor->storageOffset+i0*tensor->stride[0]+i1*tensor->stride[1]+i2*tensor->stride[2];
}

inline TYPE* TENSOR_FUNC(dataPtr4d)(TENSOR *tensor, long i0, long i1, long i2, long i3)
{
  return tensor->storage->data+tensor->storageOffset+i0*tensor->stride[0]+i1*tensor->stride[1]+i2*tensor->stride[2]+i3*tensor->stride[3];
}

inline TYPE* TENSOR_FUNC(selectPtr)(TENSOR *tensor, int dimension, long sliceIndex)
{
  return tensor->storage->data+tensor->storageOffset+sliceIndex*tensor->stride[dimension];
}

void TENSOR_FUNC(copy)(TENSOR *tensor, TENSOR *src)
{
  TH_TENSOR_APPLY2(TYPE, tensor, TYPE, src, *tensor_p = *src_p;)
}

#define THDoubleTensor THTensor

#define IMPLEMENT_TENSOR_COPY(TYPENAMESRC, TYPE_SRC) \
void TENSOR_FUNC(copy##TYPENAMESRC)(TENSOR *tensor, TH##TYPENAMESRC##Tensor *src) \
{ \
  TH_TENSOR_APPLY2(TYPE, tensor, TYPE_SRC, src, *tensor_p = (TYPE)(*src_p);) \
}

IMPLEMENT_TENSOR_COPY(Char, char)
IMPLEMENT_TENSOR_COPY(Short, short)
IMPLEMENT_TENSOR_COPY(Int, int)
IMPLEMENT_TENSOR_COPY(Long, long)
IMPLEMENT_TENSOR_COPY(Float, float)
IMPLEMENT_TENSOR_COPY(Double, double)
