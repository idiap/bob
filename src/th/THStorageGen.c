/* Storage type */
#define STORAGE_T_(TYPE) TH##TYPE##Storage
#define STORAGE_T(TYPE) STORAGE_T_(TYPE)
#define STORAGE STORAGE_T(CAP_TYPE)

/* Function name for a Storage */
#define STORAGE_FUNC_TN_(TYPE,NAME) TH##TYPE##Storage_##NAME
#define STORAGE_FUNC_TN(TYPE, NAME) STORAGE_FUNC_TN_(TYPE,NAME) 
#define STORAGE_FUNC(NAME) STORAGE_FUNC_TN(CAP_TYPE, NAME)

STORAGE* STORAGE_FUNC(new)(void)
{
  return STORAGE_FUNC(newWithSize)(0);
}

STORAGE* STORAGE_FUNC(newWithSize)(long size)
{
  STORAGE *storage = THAlloc(sizeof(STORAGE));
  storage->data = THAlloc(sizeof(TYPE)*size);
  storage->size = size;
  storage->refcount = 1;
  return storage;
}

void STORAGE_FUNC(retain)(STORAGE *storage)
{
  if(storage)
    ++storage->refcount;
}

void STORAGE_FUNC(free)(STORAGE *storage)
{
  if(!storage)
    return;

  if(storage->refcount > 0)
  {
    if(--storage->refcount == 0)
    {
      THFree(storage->data);
      THFree(storage);
    }
  }
}

void STORAGE_FUNC(resize)(STORAGE *storage, long size, int keepContent)
{
  if(keepContent)
  {
    storage->data = THRealloc(storage->data, sizeof(TYPE)*size);
    storage->size = size;
  }
  else
  {
    THFree(storage->data);
    storage->data = THAlloc(sizeof(TYPE)*size);
    storage->size = size;
  }
}

void STORAGE_FUNC(copy) (STORAGE *storage, STORAGE *src)
{
  long i;
  THArgCheck(storage->size == src->size, 2, "size mismatch");
  for(i = 0; i < storage->size; i++)
    storage->data[i] = src->data[i];
}

void STORAGE_FUNC(fill)(STORAGE *storage, TYPE value)
{
  long i;
  for(i = 0; i < storage->size; i++)
    storage->data[i] = value;
}

#define IMPLEMENT_STORAGE_COPY(TYPENAMESRC) \
void STORAGE_FUNC(copy##TYPENAMESRC)(STORAGE *storage, TH##TYPENAMESRC##Storage *src) \
{ \
  long i; \
  THArgCheck(storage->size == src->size, 2, "size mismatch"); \
  for(i = 0; i < storage->size; i++) \
    storage->data[i] = (TYPE)src->data[i]; \
}

IMPLEMENT_STORAGE_COPY(Char)
IMPLEMENT_STORAGE_COPY(Short)
IMPLEMENT_STORAGE_COPY(Int)
IMPLEMENT_STORAGE_COPY(Long)
IMPLEMENT_STORAGE_COPY(Float)
IMPLEMENT_STORAGE_COPY(Double)
