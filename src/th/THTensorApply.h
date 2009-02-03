#ifndef TH_TENSOR_APPLY_INC
#define TH_TENSOR_APPLY_INC

#define TH_TENSOR_APPLY3(TYPE1, TENSOR1, TYPE2, TENSOR2, TYPE3, TENSOR3, CODE) \
{ \
  TYPE1 *TENSOR1##_p = NULL; \
  long *TENSOR1##_counter = NULL; \
  long TENSOR1##_stride = 0, TENSOR1##_size = 0, TENSOR1##_dim = 0, TENSOR1##_i, TENSOR1##_n; \
  TYPE2 *TENSOR2##_p = NULL; \
  long *TENSOR2##_counter = NULL; \
  long TENSOR2##_stride = 0, TENSOR2##_size = 0, TENSOR2##_dim = 0, TENSOR2##_i, TENSOR2##_n; \
  TYPE2 *TENSOR3##_p = NULL; \
  long *TENSOR3##_counter = NULL; \
  long TENSOR3##_stride = 0, TENSOR3##_size = 0, TENSOR3##_dim = 0, TENSOR3##_i, TENSOR3##_n; \
  int TH_TENSOR_APPLY_hasFinished = 0; \
\
  TENSOR1##_n = (TENSOR1->nDimension ? 1 : 0); \
  for(TENSOR1##_i = 0; TENSOR1##_i < TENSOR1->nDimension; TENSOR1##_i++) \
    TENSOR1##_n *= TENSOR1->size[TENSOR1##_i]; \
\
  TENSOR2##_n = (TENSOR2->nDimension ? 1 : 0); \
  for(TENSOR2##_i = 0; TENSOR2##_i < TENSOR2->nDimension; TENSOR2##_i++) \
    TENSOR2##_n *= TENSOR2->size[TENSOR2##_i]; \
\
  TENSOR3##_n = (TENSOR3->nDimension ? 1 : 0); \
  for(TENSOR3##_i = 0; TENSOR3##_i < TENSOR3->nDimension; TENSOR3##_i++) \
    TENSOR3##_n *= TENSOR3->size[TENSOR3##_i]; \
\
  if(TENSOR1##_n != TENSOR2##_n || TENSOR1##_n != TENSOR3##_n) /* should we do the check in the function instead? i think so */ \
    THError("inconsistent tensor size"); \
\
  if(TENSOR1->nDimension == 0) \
    TH_TENSOR_APPLY_hasFinished = 1; \
  else \
  { \
    TENSOR1##_p = TENSOR1->storage->data+TENSOR1->storageOffset; \
    for(TENSOR1##_dim = 0; TENSOR1##_dim < TENSOR1->nDimension; TENSOR1##_dim++) \
    { \
      if(TENSOR1->size[TENSOR1##_dim] != 1) \
        break; \
    } \
    TENSOR1##_stride = (TENSOR1##_dim == TENSOR1->nDimension ? 0 : TENSOR1->stride[TENSOR1##_dim]); \
    TENSOR1##_size = TENSOR1->size[0]; \
    for(TENSOR1##_dim = 1; TENSOR1##_dim < TENSOR1->nDimension; TENSOR1##_dim++) \
    { \
      if(TENSOR1->size[TENSOR1##_dim] != 1) \
      { \
        if(TENSOR1->stride[TENSOR1##_dim] == TENSOR1##_size) \
          TENSOR1##_size *= TENSOR1->size[TENSOR1##_dim]; \
        else \
          break; \
      } \
    } \
    TENSOR1##_counter = (long*)THAlloc(sizeof(long)*(TENSOR1->nDimension-TENSOR1##_dim)); \
    for(TENSOR1##_i = 0; TENSOR1##_i < TENSOR1->nDimension-TENSOR1##_dim; TENSOR1##_i++) \
      TENSOR1##_counter[TENSOR1##_i] = 0; \
\
    TENSOR2##_p = TENSOR2->storage->data+TENSOR2->storageOffset; \
    for(TENSOR2##_dim = 0; TENSOR2##_dim < TENSOR2->nDimension; TENSOR2##_dim++) \
    { \
      if(TENSOR2->size[TENSOR2##_dim] != 1) \
        break; \
    } \
    TENSOR2##_stride = (TENSOR2##_dim == TENSOR2->nDimension ? 0 : TENSOR2->stride[TENSOR2##_dim]); \
    TENSOR2##_size = TENSOR2->size[0]; \
    for(TENSOR2##_dim = 1; TENSOR2##_dim < TENSOR2->nDimension; TENSOR2##_dim++) \
    { \
      if(TENSOR2->size[TENSOR2##_dim] != 1) \
      { \
        if(TENSOR2->stride[TENSOR2##_dim] == TENSOR2##_size) \
          TENSOR2##_size *= TENSOR2->size[TENSOR2##_dim]; \
        else \
          break; \
      } \
    } \
    TENSOR2##_counter = (long*)THAlloc(sizeof(long)*(TENSOR2->nDimension-TENSOR2##_dim)); \
    for(TENSOR2##_i = 0; TENSOR2##_i < TENSOR2->nDimension-TENSOR2##_dim; TENSOR2##_i++) \
      TENSOR2##_counter[TENSOR2##_i] = 0; \
\
    TENSOR3##_p = TENSOR3->storage->data+TENSOR3->storageOffset; \
    for(TENSOR3##_dim = 0; TENSOR3##_dim < TENSOR3->nDimension; TENSOR3##_dim++) \
    { \
      if(TENSOR3->size[TENSOR3##_dim] != 1) \
        break; \
    } \
    TENSOR3##_stride = (TENSOR3##_dim == TENSOR3->nDimension ? 0 : TENSOR3->stride[TENSOR3##_dim]); \
    TENSOR3##_size = TENSOR3->size[0]; \
    for(TENSOR3##_dim = 1; TENSOR3##_dim < TENSOR3->nDimension; TENSOR3##_dim++) \
    { \
      if(TENSOR3->size[TENSOR3##_dim] != 1) \
      { \
        if(TENSOR3->stride[TENSOR3##_dim] == TENSOR3##_size) \
          TENSOR3##_size *= TENSOR3->size[TENSOR3##_dim]; \
        else \
          break; \
      } \
    } \
    TENSOR3##_counter = (long*)THAlloc(sizeof(long)*(TENSOR3->nDimension-TENSOR3##_dim)); \
    for(TENSOR3##_i = 0; TENSOR3##_i < TENSOR3->nDimension-TENSOR3##_dim; TENSOR3##_i++) \
      TENSOR3##_counter[TENSOR3##_i] = 0; \
  } \
\
  TENSOR1##_i = 0; \
  TENSOR2##_i = 0; \
  TENSOR3##_i = 0; \
  while(!TH_TENSOR_APPLY_hasFinished) \
  { \
    for(; TENSOR1##_i < TENSOR1##_size && TENSOR2##_i < TENSOR2##_size && TENSOR3##_i < TENSOR3##_size; TENSOR1##_i++, TENSOR2##_i++, TENSOR3##_i++, TENSOR1##_p += TENSOR1##_stride, TENSOR2##_p += TENSOR2##_stride, TENSOR3##_p += TENSOR3##_stride) /* 0 et pas TENSOR##_dim! */ \
    { \
      CODE \
    } \
\
    if(TENSOR1##_i == TENSOR1##_size) \
    { \
      if(TENSOR1##_dim == TENSOR1->nDimension) \
         break; \
\
      TENSOR1##_p -= TENSOR1##_size*TENSOR1##_stride; \
      for(TENSOR1##_i = TENSOR1##_dim; TENSOR1##_i < TENSOR1->nDimension; TENSOR1##_i++) \
      { \
        TENSOR1##_counter[TENSOR1##_i-TENSOR1##_dim]++; \
        TENSOR1##_p += TENSOR1->stride[TENSOR1##_i]; \
\
        if(TENSOR1##_counter[TENSOR1##_i-TENSOR1##_dim]  == TENSOR1->size[TENSOR1##_i]) \
        { \
          if(TENSOR1##_i == TENSOR1->nDimension-1) \
          { \
            TH_TENSOR_APPLY_hasFinished = 1; \
            break; \
          } \
            else \
          { \
            TENSOR1##_p -= TENSOR1##_counter[TENSOR1##_i-TENSOR1##_dim]*TENSOR1->stride[TENSOR1##_i]; \
            TENSOR1##_counter[TENSOR1##_i-TENSOR1##_dim] = 0; \
          } \
        } \
        else \
          break; \
      } \
      TENSOR1##_i = 0; \
    } \
\
    if(TENSOR2##_i == TENSOR2##_size) \
    { \
      if(TENSOR2##_dim == TENSOR2->nDimension) \
         break; \
\
      TENSOR2##_p -= TENSOR2##_size*TENSOR2##_stride; \
      for(TENSOR2##_i = TENSOR2##_dim; TENSOR2##_i < TENSOR2->nDimension; TENSOR2##_i++) \
      { \
        TENSOR2##_counter[TENSOR2##_i-TENSOR2##_dim]++; \
        TENSOR2##_p += TENSOR2->stride[TENSOR2##_i]; \
\
        if(TENSOR2##_counter[TENSOR2##_i-TENSOR2##_dim]  == TENSOR2->size[TENSOR2##_i]) \
        { \
          if(TENSOR2##_i == TENSOR2->nDimension-1) \
          { \
            TH_TENSOR_APPLY_hasFinished = 1; \
            break; \
          } \
            else \
          { \
            TENSOR2##_p -= TENSOR2##_counter[TENSOR2##_i-TENSOR2##_dim]*TENSOR2->stride[TENSOR2##_i]; \
            TENSOR2##_counter[TENSOR2##_i-TENSOR2##_dim] = 0; \
          } \
        } \
        else \
          break; \
      } \
      TENSOR2##_i = 0; \
    } \
\
    if(TENSOR3##_i == TENSOR3##_size) \
    { \
      if(TENSOR3##_dim == TENSOR3->nDimension) \
         break; \
\
      TENSOR3##_p -= TENSOR3##_size*TENSOR3##_stride; \
      for(TENSOR3##_i = TENSOR3##_dim; TENSOR3##_i < TENSOR3->nDimension; TENSOR3##_i++) \
      { \
        TENSOR3##_counter[TENSOR3##_i-TENSOR3##_dim]++; \
        TENSOR3##_p += TENSOR3->stride[TENSOR3##_i]; \
\
        if(TENSOR3##_counter[TENSOR3##_i-TENSOR3##_dim]  == TENSOR3->size[TENSOR3##_i]) \
        { \
          if(TENSOR3##_i == TENSOR3->nDimension-1) \
          { \
            TH_TENSOR_APPLY_hasFinished = 1; \
            break; \
          } \
            else \
          { \
            TENSOR3##_p -= TENSOR3##_counter[TENSOR3##_i-TENSOR3##_dim]*TENSOR3->stride[TENSOR3##_i]; \
            TENSOR3##_counter[TENSOR3##_i-TENSOR3##_dim] = 0; \
          } \
        } \
        else \
          break; \
      } \
      TENSOR3##_i = 0; \
    } \
  } \
  THFree(TENSOR1##_counter); \
  THFree(TENSOR2##_counter); \
  THFree(TENSOR3##_counter); \
}

#define TH_TENSOR_APPLY2(TYPE1, TENSOR1, TYPE2, TENSOR2, CODE) \
{ \
  TYPE1 *TENSOR1##_p = NULL; \
  long *TENSOR1##_counter = NULL; \
  long TENSOR1##_stride = 0, TENSOR1##_size = 0, TENSOR1##_dim = 0, TENSOR1##_i, TENSOR1##_n; \
  TYPE2 *TENSOR2##_p = NULL; \
  long *TENSOR2##_counter = NULL; \
  long TENSOR2##_stride = 0, TENSOR2##_size = 0, TENSOR2##_dim = 0, TENSOR2##_i, TENSOR2##_n; \
  int TH_TENSOR_APPLY_hasFinished = 0; \
\
  TENSOR1##_n = (TENSOR1->nDimension ? 1 : 0); \
  for(TENSOR1##_i = 0; TENSOR1##_i < TENSOR1->nDimension; TENSOR1##_i++) \
    TENSOR1##_n *= TENSOR1->size[TENSOR1##_i]; \
\
  TENSOR2##_n = (TENSOR2->nDimension ? 1 : 0); \
  for(TENSOR2##_i = 0; TENSOR2##_i < TENSOR2->nDimension; TENSOR2##_i++) \
    TENSOR2##_n *= TENSOR2->size[TENSOR2##_i]; \
\
  if(TENSOR1##_n != TENSOR2##_n) /* should we do the check in the function instead? i think so */ \
    THError("inconsistent tensor size"); \
\
  if(TENSOR1->nDimension == 0) \
    TH_TENSOR_APPLY_hasFinished = 1; \
  else \
  { \
    TENSOR1##_p = TENSOR1->storage->data+TENSOR1->storageOffset; \
    for(TENSOR1##_dim = 0; TENSOR1##_dim < TENSOR1->nDimension; TENSOR1##_dim++) \
    { \
      if(TENSOR1->size[TENSOR1##_dim] != 1) \
        break; \
    } \
    TENSOR1##_stride = (TENSOR1##_dim == TENSOR1->nDimension ? 0 : TENSOR1->stride[TENSOR1##_dim]); \
    TENSOR1##_size = TENSOR1->size[0]; \
    for(TENSOR1##_dim = 1; TENSOR1##_dim < TENSOR1->nDimension; TENSOR1##_dim++) \
    { \
      if(TENSOR1->size[TENSOR1##_dim] != 1) \
      { \
        if(TENSOR1->stride[TENSOR1##_dim] == TENSOR1##_size) \
          TENSOR1##_size *= TENSOR1->size[TENSOR1##_dim]; \
        else \
          break; \
      } \
    } \
    TENSOR1##_counter = (long*)THAlloc(sizeof(long)*(TENSOR1->nDimension-TENSOR1##_dim)); \
    for(TENSOR1##_i = 0; TENSOR1##_i < TENSOR1->nDimension-TENSOR1##_dim; TENSOR1##_i++) \
      TENSOR1##_counter[TENSOR1##_i] = 0; \
\
    TENSOR2##_p = TENSOR2->storage->data+TENSOR2->storageOffset; \
    for(TENSOR2##_dim = 0; TENSOR2##_dim < TENSOR2->nDimension; TENSOR2##_dim++) \
    { \
      if(TENSOR2->size[TENSOR2##_dim] != 1) \
        break; \
    } \
    TENSOR2##_stride = (TENSOR2##_dim == TENSOR2->nDimension ? 0 : TENSOR2->stride[TENSOR2##_dim]); \
    TENSOR2##_size = TENSOR2->size[0]; \
    for(TENSOR2##_dim = 1; TENSOR2##_dim < TENSOR2->nDimension; TENSOR2##_dim++) \
    { \
      if(TENSOR2->size[TENSOR2##_dim] != 1) \
      { \
        if(TENSOR2->stride[TENSOR2##_dim] == TENSOR2##_size) \
          TENSOR2##_size *= TENSOR2->size[TENSOR2##_dim]; \
        else \
          break; \
      } \
    } \
    TENSOR2##_counter = (long*)THAlloc(sizeof(long)*(TENSOR2->nDimension-TENSOR2##_dim)); \
    for(TENSOR2##_i = 0; TENSOR2##_i < TENSOR2->nDimension-TENSOR2##_dim; TENSOR2##_i++) \
      TENSOR2##_counter[TENSOR2##_i] = 0; \
  } \
\
  TENSOR1##_i = 0; \
  TENSOR2##_i = 0; \
  while(!TH_TENSOR_APPLY_hasFinished) \
  { \
    for(; TENSOR1##_i < TENSOR1##_size && TENSOR2##_i < TENSOR2##_size; TENSOR1##_i++, TENSOR2##_i++, TENSOR1##_p += TENSOR1##_stride, TENSOR2##_p += TENSOR2##_stride) /* 0 et pas TENSOR##_dim! */ \
    { \
      CODE \
    } \
\
    if(TENSOR1##_i == TENSOR1##_size) \
    { \
      if(TENSOR1##_dim == TENSOR1->nDimension) \
         break; \
\
      TENSOR1##_p -= TENSOR1##_size*TENSOR1##_stride; \
      for(TENSOR1##_i = TENSOR1##_dim; TENSOR1##_i < TENSOR1->nDimension; TENSOR1##_i++) \
      { \
        TENSOR1##_counter[TENSOR1##_i-TENSOR1##_dim]++; \
        TENSOR1##_p += TENSOR1->stride[TENSOR1##_i]; \
\
        if(TENSOR1##_counter[TENSOR1##_i-TENSOR1##_dim]  == TENSOR1->size[TENSOR1##_i]) \
        { \
          if(TENSOR1##_i == TENSOR1->nDimension-1) \
          { \
            TH_TENSOR_APPLY_hasFinished = 1; \
            break; \
          } \
            else \
          { \
            TENSOR1##_p -= TENSOR1##_counter[TENSOR1##_i-TENSOR1##_dim]*TENSOR1->stride[TENSOR1##_i]; \
            TENSOR1##_counter[TENSOR1##_i-TENSOR1##_dim] = 0; \
          } \
        } \
        else \
          break; \
      } \
      TENSOR1##_i = 0; \
    } \
\
    if(TENSOR2##_i == TENSOR2##_size) \
    { \
      if(TENSOR2##_dim == TENSOR2->nDimension) \
         break; \
\
      TENSOR2##_p -= TENSOR2##_size*TENSOR2##_stride; \
      for(TENSOR2##_i = TENSOR2##_dim; TENSOR2##_i < TENSOR2->nDimension; TENSOR2##_i++) \
      { \
        TENSOR2##_counter[TENSOR2##_i-TENSOR2##_dim]++; \
        TENSOR2##_p += TENSOR2->stride[TENSOR2##_i]; \
\
        if(TENSOR2##_counter[TENSOR2##_i-TENSOR2##_dim]  == TENSOR2->size[TENSOR2##_i]) \
        { \
          if(TENSOR2##_i == TENSOR2->nDimension-1) \
          { \
            TH_TENSOR_APPLY_hasFinished = 1; \
            break; \
          } \
            else \
          { \
            TENSOR2##_p -= TENSOR2##_counter[TENSOR2##_i-TENSOR2##_dim]*TENSOR2->stride[TENSOR2##_i]; \
            TENSOR2##_counter[TENSOR2##_i-TENSOR2##_dim] = 0; \
          } \
        } \
        else \
          break; \
      } \
      TENSOR2##_i = 0; \
    } \
  } \
  THFree(TENSOR1##_counter); \
  THFree(TENSOR2##_counter); \
}

#define TH_TENSOR_APPLY(TYPE, TENSOR, CODE) \
{ \
  TYPE *TENSOR##_p = NULL; \
  long *TENSOR##_counter = NULL; \
  long TENSOR##_stride = 0, TENSOR##_size = 0, TENSOR##_dim = 0, TENSOR##_i; \
  int TH_TENSOR_APPLY_hasFinished = 0; \
\
  if(TENSOR->nDimension == 0) \
    TH_TENSOR_APPLY_hasFinished = 1; \
  else \
  { \
    TENSOR##_p = TENSOR->storage->data+TENSOR->storageOffset; \
    for(TENSOR##_dim = 0; TENSOR##_dim < TENSOR->nDimension; TENSOR##_dim++) \
    { \
      if(TENSOR->size[TENSOR##_dim] != 1) \
        break; \
    } \
    TENSOR##_stride = (TENSOR##_dim == TENSOR->nDimension ? 0 : TENSOR->stride[TENSOR##_dim]); \
    TENSOR##_size = TENSOR->size[0]; \
    for(TENSOR##_dim = 1; TENSOR##_dim < TENSOR->nDimension; TENSOR##_dim++) \
    { \
      if(TENSOR->size[TENSOR##_dim] != 1) \
      { \
        if(TENSOR->stride[TENSOR##_dim] == TENSOR##_size) \
          TENSOR##_size *= TENSOR->size[TENSOR##_dim]; \
        else \
          break; \
      } \
    } \
    TENSOR##_counter = (long*)THAlloc(sizeof(long)*(TENSOR->nDimension-TENSOR##_dim)); \
    for(TENSOR##_i = 0; TENSOR##_i < TENSOR->nDimension-TENSOR##_dim; TENSOR##_i++) \
      TENSOR##_counter[TENSOR##_i] = 0; \
  } \
\
  while(!TH_TENSOR_APPLY_hasFinished) \
  { \
    for(TENSOR##_i = 0; TENSOR##_i < TENSOR##_size; TENSOR##_i++, TENSOR##_p += TENSOR##_stride) /* 0 et pas TENSOR##_dim! */ \
    { \
      CODE \
    } \
\
    if(TENSOR##_dim == TENSOR->nDimension) \
       break; \
 \
    TENSOR##_p -= TENSOR##_i*TENSOR##_stride; \
    for(TENSOR##_i = TENSOR##_dim; TENSOR##_i < TENSOR->nDimension; TENSOR##_i++) \
    { \
      TENSOR##_counter[TENSOR##_i-TENSOR##_dim]++; \
      TENSOR##_p += TENSOR->stride[TENSOR##_i]; \
\
      if(TENSOR##_counter[TENSOR##_i-TENSOR##_dim]  == TENSOR->size[TENSOR##_i]) \
      { \
        if(TENSOR##_i == TENSOR->nDimension-1) \
        { \
          TH_TENSOR_APPLY_hasFinished = 1; \
          break; \
        } \
        else \
        { \
          TENSOR##_p -= TENSOR##_counter[TENSOR##_i-TENSOR##_dim]*TENSOR->stride[TENSOR##_i]; \
          TENSOR##_counter[TENSOR##_i-TENSOR##_dim] = 0; \
        } \
      } \
      else \
        break; \
    } \
  } \
  THFree(TENSOR##_counter); \
}

#endif
