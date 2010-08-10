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
  storage->isMapped = 0;
  return storage;
}

#if defined(_WIN32) || defined(HAVE_MMAP)

STORAGE* STORAGE_FUNC(newWithMapping)(const char *fileName, int isShared)
{
  STORAGE *storage = THAlloc(sizeof(STORAGE));
  long size;

  /* check size */
  FILE *f = fopen(fileName, "rb");
  if(f == NULL)
    THError("unable to open file <%s> for mapping (read-only mode)", fileName);
  fseek(f, 0, SEEK_END);
  size = ftell(f);
  fclose(f);
  size /= sizeof(TYPE);

#ifdef _WIN32
  {
    HANDLE hfile;
    HANDLE hmfile;
    DWORD size_hi, size_lo;

    /* open file */
    if(isShared)
    {
      hfile = CreateFileA(fileName, GENERIC_READ|GENERIC_WRITE, FILE_SHARE_WRITE|FILE_SHARE_READ, 0, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, 0);
      if (hfile == INVALID_HANDLE_VALUE)
        THError("could not open file <%s> in read-write mode", fileName);
    }
    else
    {
      hfile = CreateFileA(fileName, GENERIC_READ, FILE_SHARE_WRITE|FILE_SHARE_READ, 0, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, 0);
      if (hfile == INVALID_HANDLE_VALUE)
        THError("could not open file <%s> in read-only mode", fileName);
    }

#if SIZEOF_SIZE_T > 4
    size_hi = (DWORD)((size*sizeof(TYPE)) >> 32);
    size_lo = (DWORD)((size*sizeof(TYPE)) & 0xFFFFFFFF);
#else
    size_hi = 0;
    size_lo = (DWORD)(size*sizeof(TYPE));
#endif

    /* get map handle */
    if(isShared)
    {
      if( (hmfile = CreateFileMapping(hfile, NULL, PAGE_READWRITE, size_hi, size_lo, NULL)) == NULL )
        THError("could not create a map on file <%s>", fileName);
    }
    else
    {
      if( (hmfile = CreateFileMapping(hfile, NULL, PAGE_WRITECOPY, size_hi, size_lo, NULL)) == NULL )
        THError("could not create a map on file <%s>", fileName);
    }

    /* map the stuff */
    storage = STORAGE_FUNC(new)();
    if(isShared)
      storage->data = MapViewOfFile(hmfile, FILE_MAP_ALL_ACCESS, 0, 0, 0);
    else
      storage->data = MapViewOfFile(hmfile, FILE_MAP_COPY, 0, 0, 0);
      
    storage->size = size;
    if(storage->data == NULL)
    {
      STORAGE_FUNC(free)(storage);
      THError("memory map failed on file <%s>", fileName);
    }
    CloseHandle(hfile); 
    CloseHandle(hmfile); 
  }
#else
  {
    /* open file */
    int fd;
    if(isShared)
    {
      fd = open(fileName, O_RDWR);
      if(fd == -1)
        THError("unable to open file <%s> in read-write mode", fileName);
    }
    else
    {
      fd = open(fileName, O_RDONLY);
      if(fd == -1)
        THError("unable to open file <%s> in read-only mode", fileName);
    }
    
    /* map it */
    storage = STORAGE_FUNC(new)();
    if(isShared)
      storage->data = mmap(NULL, size*sizeof(TYPE), PROT_READ|PROT_WRITE, MAP_SHARED, fd, 0);
    else
      storage->data = mmap(NULL, size*sizeof(TYPE), PROT_READ|PROT_WRITE, MAP_PRIVATE, fd, 0);

    storage->size = size;
    if(storage->data == MAP_FAILED)
    {
      storage->data = NULL; /* let's be sure it is NULL before calling free() */
      STORAGE_FUNC(free)(storage);
      THError("memory map failed on file <%s>", fileName);
    }
    close (fd);
  }
#endif

  storage->refcount = 1;
  storage->isMapped = 1;
  return storage;
}

#else

STORAGE* STORAGE_FUNC(newWithMapping)(const char *fileName, int isShared)
{
  THError("Mapped file Storages are not supported on your system");
}

#endif

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
#if defined(_WIN32) || defined(HAVE_MMAP)
      if(storage->isMapped)
      {
#ifdef _WIN32
        if(!UnmapViewOfFile((LPINT)storage->data))
#else
        if (munmap(storage->data, storage->size*sizeof(TYPE)))
#endif
          THError("could not unmap the shared memory file");
      }
      else
#endif
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
