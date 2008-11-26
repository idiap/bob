/* Storage type */
#define STORAGE_T_(TYPE) TH##TYPE##Storage
#define STORAGE_T(TYPE) STORAGE_T_(TYPE)
#define STORAGE STORAGE_T(CAP_TYPE)

/* Tensor type */
#define TENSOR_T_(TYPE) TH##TYPE##Tensor
#define TENSOR_T(TYPE) TENSOR_T_(TYPE)
#define TENSOR TENSOR_T(CAP_TYPE)

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

typedef struct TENSOR_FUNC(_)
{
    long *size;
    long *stride;
    int nDimension;
    
    STORAGE *storage;
    long storageOffset;
    int ownStorage;
    int refcount;

} TENSOR;

TH_API TENSOR *TENSOR_FUNC(new)(void);
TH_API TENSOR *TENSOR_FUNC(newWithTensor)(TENSOR *tensor);
/* stride might be NULL */
TH_API TENSOR *TENSOR_FUNC(newWithStorage)(STORAGE *storage_, long storageOffset_, int nDimension, long *size_, long *stride_);
TH_API TENSOR *TENSOR_FUNC(newWithStorage1d)(STORAGE *storage_, long storageOffset_,
                                long size0_, long stride0_);
TH_API TENSOR *TENSOR_FUNC(newWithStorage2d)(STORAGE *storage_, long storageOffset_,
                                long size0_, long stride0_,
                                long size1_, long stride1_);
TH_API TENSOR *TENSOR_FUNC(newWithStorage3d)(STORAGE *storage_, long storageOffset_,
                                long size0_, long stride0_,
                                long size1_, long stride1_,
                                long size2_, long stride2_);
TH_API TENSOR *TENSOR_FUNC(newWithStorage4d)(STORAGE *storage_, long storageOffset_,
                                long size0_, long stride0_,
                                long size1_, long stride1_,
                                long size2_, long stride2_,
                                long size3_, long stride3_);


/* stride might be NULL */
TH_API TENSOR *TENSOR_FUNC(newWithSize)(int nDimension, long *size_, long *stride_);
TH_API TENSOR *TENSOR_FUNC(newWithSize1d)(long size0_);
TH_API TENSOR *TENSOR_FUNC(newWithSize2d)(long size0_, long size1_);
TH_API TENSOR *TENSOR_FUNC(newWithSize3d)(long size0_, long size1_, long size2_);
TH_API TENSOR *TENSOR_FUNC(newWithSize4d)(long size0_, long size1_, long size2_, long size3_);

TH_API void TENSOR_FUNC(setTensor)(TENSOR *tensor, TENSOR *src);
/* stride might be NULL */
TH_API void TENSOR_FUNC(setStorage)(TENSOR *tensor, STORAGE *storage_, long storageOffset_, int nDimension, long *size_, long *stride_);
TH_API void TENSOR_FUNC(setStorage1d)(TENSOR *tensor, STORAGE *storage_, long storageOffset_,
                         long size0_, long stride0_);
TH_API void TENSOR_FUNC(setStorage2d)(TENSOR *tensor, STORAGE *storage_, long storageOffset_,
                         long size0_, long stride0_,
                         long size1_, long stride1_);
TH_API void TENSOR_FUNC(setStorage3d)(TENSOR *tensor, STORAGE *storage_, long storageOffset_,
                         long size0_, long stride0_,
                         long size1_, long stride1_,
                         long size2_, long stride2_);
TH_API void TENSOR_FUNC(setStorage4d)(TENSOR *tensor, STORAGE *storage_, long storageOffset_,
                         long size0_, long stride0_,
                         long size1_, long stride1_,
                         long size2_, long stride2_,
                         long size3_, long stride3_);

/* Slow access methods [check everything] */
TH_API void TENSOR_FUNC(set1d)(TENSOR *tensor, long x0, TYPE value);
TH_API void TENSOR_FUNC(set2d)(TENSOR *tensor, long x0, long x1, TYPE value);
TH_API void TENSOR_FUNC(set3d)(TENSOR *tensor, long x0, long x1, long x2, TYPE value);
TH_API void TENSOR_FUNC(set4d)(TENSOR *tensor, long x0, long x1, long x2, long x3, TYPE value);

TH_API TYPE TENSOR_FUNC(get1d)(TENSOR *tensor, long x0);
TH_API TYPE TENSOR_FUNC(get2d)(TENSOR *tensor, long x0, long x1);
TH_API TYPE TENSOR_FUNC(get3d)(TENSOR *tensor, long x0, long x1, long x2);
TH_API TYPE TENSOR_FUNC(get4d)(TENSOR *tensor, long x0, long x1, long x2, long x3);
  
TH_API void TENSOR_FUNC(resizeAs)(TENSOR *tensor, TENSOR *src);
TH_API void TENSOR_FUNC(resize)(TENSOR *tensor, int nDimension, long *size_);
TH_API void TENSOR_FUNC(resize1d)(TENSOR *tensor, long size0_);
TH_API void TENSOR_FUNC(resize2d)(TENSOR *tensor, long size0_, long size1_);
TH_API void TENSOR_FUNC(resize3d)(TENSOR *tensor, long size0_, long size1_, long size2_);
TH_API void TENSOR_FUNC(resize4d)(TENSOR *tensor, long size0_, long size1_, long size2_, long size3_);

/* je me demande si on devrait pas modifier soit-meme: ca ferait un malloc pour le new, mais plus clair, non? */
TH_API void TENSOR_FUNC(narrow)(TENSOR *tensor, TENSOR *src, int dimension_, long firstIndex_, long size_);
TH_API void TENSOR_FUNC(select)(TENSOR *tensor, TENSOR *src, int dimension_, long sliceIndex_);

TH_API void TENSOR_FUNC(transpose)(TENSOR *tensor, TENSOR *src, int dimension1_, int dimension2_);

TH_API void TENSOR_FUNC(unfold)(TENSOR *tensor, TENSOR *src, int dimension_, long size_, long step_);
    
TH_API void TENSOR_FUNC(fill)(TENSOR *tensor, TYPE value);
TH_API int TENSOR_FUNC(isContiguous)(TENSOR *tensor);
TH_API long TENSOR_FUNC(nElement)(TENSOR *tensor);

TH_API void TENSOR_FUNC(retain)(TENSOR *tensor);
TH_API void TENSOR_FUNC(free)(TENSOR *tensor);

TH_API inline TYPE* TENSOR_FUNC(dataPtr)(TENSOR *tensor);
TH_API inline TYPE* TENSOR_FUNC(dataPtr1d)(TENSOR *tensor, long i0);
TH_API inline TYPE* TENSOR_FUNC(dataPtr2d)(TENSOR *tensor, long i0, long i1);
TH_API inline TYPE* TENSOR_FUNC(dataPtr3d)(TENSOR *tensor, long i0, long i1, long i2);
TH_API inline TYPE* TENSOR_FUNC(dataPtr4d)(TENSOR *tensor, long i0, long i1, long i2, long i3);
TH_API inline TYPE* TENSOR_FUNC(selectPtr)(TENSOR *tensor, int dimension, long sliceIndex);

/* Support for copy between different Tensor types */

struct THCharTensor__;
struct THShortTensor__;
struct THIntTensor__;
struct THLongTensor__;
struct THFloatTensor__;
struct THTensor__;

TH_API void TENSOR_FUNC(copy)(TENSOR *tensor, TENSOR *src);
TH_API void TENSOR_FUNC(copyChar)(TENSOR *tensor, struct THCharTensor__ *src);
TH_API void TENSOR_FUNC(copyShort)(TENSOR *tensor, struct THShortTensor__ *src);
TH_API void TENSOR_FUNC(copyInt)(TENSOR *tensor, struct THIntTensor__ *src);
TH_API void TENSOR_FUNC(copyLong)(TENSOR *tensor, struct THLongTensor__ *src);
TH_API void TENSOR_FUNC(copyFloat)(TENSOR *tensor, struct THFloatTensor__ *src);
TH_API void TENSOR_FUNC(copyDouble)(TENSOR *tensor, struct THTensor__ *src);

#undef STORAGE_T_
#undef STORAGE_T
#undef STORAGE
#undef TENSOR_T_
#undef TENSOR_T
#undef TENSOR
#undef TENSOR_FUNC_TN_
#undef TENSOR_FUNC_TN
#undef TENSOR_FUNC
