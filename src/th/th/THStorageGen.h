/* Storage type */
#define STORAGE_T_(TYPE) TH##TYPE##Storage
#define STORAGE_T(TYPE) STORAGE_T_(TYPE)
#define STORAGE STORAGE_T(CAP_TYPE)

/* Function name for a Storage */
#define STORAGE_FUNC_TN_(TYPE,NAME) TH##TYPE##Storage_##NAME
#define STORAGE_FUNC_TN(TYPE, NAME) STORAGE_FUNC_TN_(TYPE,NAME) 
#define STORAGE_FUNC(NAME) STORAGE_FUNC_TN(CAP_TYPE, NAME)

typedef struct STORAGE_FUNC(_)
{
    TYPE *data;
    long size;
    int refcount;
    char isMapped;
} STORAGE;

TH_API STORAGE* STORAGE_FUNC(new)(void);
TH_API STORAGE* STORAGE_FUNC(newWithSize)(long size);
TH_API STORAGE* STORAGE_FUNC(newWithMapping)(const char *fileName, int isShared);
TH_API void STORAGE_FUNC(retain)(STORAGE *storage);
TH_API void STORAGE_FUNC(free)(STORAGE *storage);
TH_API void STORAGE_FUNC(resize)(STORAGE *storage, long size, int keepContent);
TH_API void STORAGE_FUNC(copy)(STORAGE *storage, STORAGE *src);
TH_API void STORAGE_FUNC(fill)(STORAGE *storage, TYPE value);

/* Support for copy between different Storage types */

struct THCharStorage__;
struct THShortStorage__;
struct THIntStorage__;
struct THLongStorage__;
struct THFloatStorage__;
struct THDoubleStorage__;

TH_API void STORAGE_FUNC(copyChar)(STORAGE *storage, struct THCharStorage__ *src);
TH_API void STORAGE_FUNC(copyShort)(STORAGE *storage, struct THShortStorage__ *src);
TH_API void STORAGE_FUNC(copyInt)(STORAGE *storage, struct THIntStorage__ *src);
TH_API void STORAGE_FUNC(copyLong)(STORAGE *storage, struct THLongStorage__ *src);
TH_API void STORAGE_FUNC(copyFloat)(STORAGE *storage, struct THFloatStorage__ *src);
TH_API void STORAGE_FUNC(copyDouble)(STORAGE *storage, struct THDoubleStorage__ *src);

#undef STORAGE_T_
#undef STORAGE_T
#undef STORAGE
#undef STORAGE_FUNC_TN_
#undef STORAGE_FUNC_TN
#undef STORAGE_FUNC
