/* Tensor type */
#define TENSOR_T_(TYPE) TH##TYPE##Tensor
#define TENSOR_T(TYPE) TENSOR_T_(TYPE)
#define TENSOR TENSOR_T(CAP_TYPE)

/* Function name for a Tensor */
#define TENSOR_FUNC_TN_(TYPE,NAME) TH##TYPE##Tensor_##NAME
#define TENSOR_FUNC_TN(TYPE, NAME) TENSOR_FUNC_TN_(TYPE,NAME)
#define TENSOR_FUNC(NAME) TENSOR_FUNC_TN(CAP_TYPE, NAME)

/* Class name for a Tensor */
#define TENSOR_CLASS_TN_(TYPE,NAME) TYPE##NAME
#define TENSOR_CLASS_TN(TYPE, NAME) TENSOR_CLASS_TN_(TYPE,NAME)
#define TENSOR_CLASS(NAME) TENSOR_CLASS_TN(CAP_TYPE, NAME)

/* For the default Tensor type, we simplify the naming */
#ifdef DEFAULT_TENSOR
#undef TENSOR
#undef TENSOR_FUNC
//#undef TENSOR_CLASS
#define TENSOR THTensor
#define TENSOR_FUNC(NAME) TENSOR_FUNC_TN(, NAME)
//#define TENSOR_CLASS(NAME) TENSOR_CLASS_TN(, NAME)
#endif

TENSOR_CLASS(Tensor)::TENSOR_CLASS(Tensor)() : Tensor(DATATYPE)
{
  t = TENSOR_FUNC(new)();
}

TENSOR_CLASS(Tensor)::TENSOR_CLASS(Tensor)(long dim0_) : Tensor(DATATYPE)
{
  t = TENSOR_FUNC(newWithSize1d)(dim0_);
}

TENSOR_CLASS(Tensor)::TENSOR_CLASS(Tensor)(long dim0_, long dim1_) : Tensor(DATATYPE)
{
  t = TENSOR_FUNC(newWithSize2d)(dim0_, dim1_);
}

TENSOR_CLASS(Tensor)::TENSOR_CLASS(Tensor)(long dim0_, long dim1_, long dim2_) : Tensor(DATATYPE)
{
  t = TENSOR_FUNC(newWithSize3d)(dim0_, dim1_, dim2_);
}

TENSOR_CLASS(Tensor)::TENSOR_CLASS(Tensor)(long dim0_, long dim1_, long dim2_, long dim3_) : Tensor(DATATYPE)
{
  t = TENSOR_FUNC(newWithSize4d)(dim0_, dim1_, dim2_, dim3_);
}

void TENSOR_CLASS(Tensor)::setTensor(const Tensor *src_)
{
  short datatype_ = src_->getDatatype();

  if(getDatatype() == datatype_)
    {
      const TENSOR_CLASS(Tensor) *src = (const TENSOR_CLASS(Tensor) *) src_;
      TENSOR_FUNC(free)(t);
      t = TENSOR_FUNC(new)();
      TENSOR_FUNC(setTensor)(t, src->t);
    }
  else
    {
      print("Error: TENSOR_CLASS(Tensor)::setTensor() don't know how to set a Tensor from a different type. Try a copy instead.");
    }
}

void TENSOR_CLASS(Tensor)::copy(const Tensor *src_)
{
  short datatype_ = src_->getDatatype();

  switch(datatype_)
    {
    case 0: // Char
      //CharTensor *csrc = (CharTensor *) src_;
      //TENSOR_FUNC(copyChar)(t, csrc->t);
      TENSOR_FUNC(copyChar)(t, ((const CharTensor *) src_)->t);
      break;

    case 1: // Short
      //ShortTensor *ssrc = (ShortTensor *)src_;
      //TENSOR_FUNC(copyShort)(t, ssrc->t);
      TENSOR_FUNC(copyShort)(t, ((const ShortTensor *) src_)->t);
      break;

    case 2: // Int
      //IntTensor *isrc = (IntTensor *) src_;
      //TENSOR_FUNC(copyInt)(t, isrc->t);
      TENSOR_FUNC(copyInt)(t, ((const IntTensor *) src_)->t);
      break;

    case 3: // Long
      //LongTensor *lsrc = (LongTensor *) src_;
      //TENSOR_FUNC(copyLong)(t, lsrc->t);
      TENSOR_FUNC(copyLong)(t, ((const LongTensor *) src_)->t);
      break;

    case 4: // Float
      //FloatTensor *fsrc = (FloatTensor *) src_;
      //TENSOR_FUNC(copyFloat)(t, fsrc->t);
      TENSOR_FUNC(copyFloat)(t, ((const FloatTensor *) src_)->t);
      break;

    case 5: // Double
      //DoubleTensor *dsrc = (DoubleTensor *) src_;
      //TENSOR_FUNC(copyDouble)(t, dsrc->t);
      TENSOR_FUNC(copyDouble)(t, ((const DoubleTensor *) src_)->t);
      break;
    }
}

void TENSOR_CLASS(Tensor)::transpose(const Tensor *src, int dimension1_, int dimension2_)
{
  short datatype_ = src->getDatatype();

  if(getDatatype() == datatype_)
    {
      TENSOR_CLASS(Tensor) *src_ = (TENSOR_CLASS(Tensor) *) src;
      TENSOR_FUNC(transpose)(t, src_->t, dimension1_, dimension2_);
    }
  else
    {
      print("Error: TENSOR_CLASS(Tensor)::transpose() don't know how to transpose a Tensor from a different type. Do a copy first.");
    }
}

void TENSOR_CLASS(Tensor)::narrow(const Tensor *src, int dimension_, long firstIndex_, long size_)
{
  short datatype_ = src->getDatatype();

  if(getDatatype() == datatype_)
    {
      TENSOR_CLASS(Tensor) *src_ = (TENSOR_CLASS(Tensor) *) src;
      TENSOR_FUNC(narrow)(t, src_->t, dimension_, firstIndex_, size_);
    }
  else
    {
       print("Error: TENSOR_CLASS(Tensor)::narrow() don't know how to transpose a Tensor from a different type. Do a copy first.");
    }
}

void TENSOR_CLASS(Tensor)::select(const Tensor *src, int dimension_, long sliceIndex_)
{
  short datatype_ = src->getDatatype();

  if(getDatatype() == datatype_)
    {
      TENSOR_CLASS(Tensor) *src_ = (TENSOR_CLASS(Tensor) *) src;
      TENSOR_FUNC(select)(t, src_->t, dimension_, sliceIndex_);
    }
  else
    {
      print("Error: TENSOR_CLASS(Tensor)::select() don't know how to transpose a Tensor from a different type. Do a copy first.");
    }
}

Tensor* TENSOR_CLASS(Tensor)::select(int dimension_, long sliceIndex_) const
{
	TENSOR_CLASS(Tensor) *dst = new TENSOR_CLASS(Tensor)();
	TENSOR_FUNC(select)(dst->t, t, dimension_, sliceIndex_);
	return (Tensor*)dst;
}

void TENSOR_CLASS(Tensor)::unfold(const Tensor *src, int dimension_, long size_, long step_)
{
  short datatype_ = src->getDatatype();

  if(getDatatype() == datatype_)
    {
      TENSOR_CLASS(Tensor) *src_ = (TENSOR_CLASS(Tensor) *) src;
      TENSOR_FUNC(unfold)(t, src_->t, dimension_, size_, step_);
    }
  else
    {
      error("TENSOR_CLASS(Tensor)::unfold() don't know how to unfold a Tensor from a different type. Do a copy first.");
    }
}

void TENSOR_CLASS(Tensor)::print(const char *name) const
{
  if(name != NULL) Torch::print("Tensor %s:\n", name);
  Tprint((TENSOR_CLASS(Tensor) *)this);
}

void TENSOR_CLASS(Tensor)::sprint(const char *name, ...) const
{
  if(name != NULL)
  {
	char _msg[512];

	va_list args;
	va_start(args, name);
	vsprintf(_msg, name, args);
  	Torch::print("Tensor %s:\n", _msg);
	fflush(stdout);
	va_end(args);
  }
  Tprint((TENSOR_CLASS(Tensor) *)this);
}

void TENSOR_CLASS(Tensor)::fill(TYPE value)
{
  TENSOR_FUNC(fill)(t, value);
}

void TENSOR_CLASS(Tensor)::resize(long dim0) const
{
  TENSOR_FUNC(resize4d)(t, dim0, -1, -1, -1);
}

void TENSOR_CLASS(Tensor)::resize(long dim0, long dim1) const
{
  TENSOR_FUNC(resize4d)(t, dim0, dim1, -1, -1);
}

void TENSOR_CLASS(Tensor)::resize(long dim0, long dim1, long dim2) const
{
  TENSOR_FUNC(resize4d)(t, dim0, dim1, dim2, -1);
}

void TENSOR_CLASS(Tensor)::resize(long dim0, long dim1, long dim2, long dim3) const
{
  TENSOR_FUNC(resize4d)(t, dim0, dim1, dim2, dim3);
}

TYPE TENSOR_CLASS(Tensor)::get(long x0) const
{
  TYPE v = TENSOR_FUNC(get1d)(t, x0);
  // (t->storage->data+t->storageOffset)[x0*t->stride[0]]
  return v;
}

TYPE TENSOR_CLASS(Tensor)::get(long x0, long x1) const
{
  TYPE v = TENSOR_FUNC(get2d)(t, x0, x1);
  // (t->storage->data+t->storageOffset)[x0*t->stride[0]+x1*t->stride[1]]
  return v;
}

TYPE TENSOR_CLASS(Tensor)::get(long x0, long x1, long x2) const
{
  TYPE v = TENSOR_FUNC(get3d)(t, x0, x1, x2);
  // (t->storage->data+t->storageOffset)[x0*t->stride[0]+x1*t->stride[1]+x2*t->stride[2]]
  return v;
}

TYPE TENSOR_CLASS(Tensor)::get(long x0, long x1, long x2, long x3) const
{
  TYPE v = TENSOR_FUNC(get4d)(t, x0, x1, x2, x3);
  // (t->storage->data+t->storageOffset)[x0*t->stride[0]+x1*t->stride[1]+x2*t->stride[2]+x3*t->stride[3]]
  return v;
}


void TENSOR_CLASS(Tensor)::setValue(long x0, TYPE v)
{
  TENSOR_FUNC(set1d)(t, x0, v);
}

void TENSOR_CLASS(Tensor)::setValue(long x0, long x1, TYPE v)
{
  TENSOR_FUNC(set2d)(t, x0, x1, v);
}

void TENSOR_CLASS(Tensor)::setValue(long x0, long x1, long x2, TYPE v)
{
  TENSOR_FUNC(set3d)(t, x0, x1, x2, v);
}

void TENSOR_CLASS(Tensor)::setValue(long x0, long x1, long x2, long x3, TYPE v)
{
  TENSOR_FUNC(set4d)(t, x0, x1, x2, x3, v);
}

TYPE &TENSOR_CLASS(Tensor)::operator()(long x0)
{
  return (t->storage->data+t->storageOffset)[x0*t->stride[0]];
}

TYPE &TENSOR_CLASS(Tensor)::operator()(long x0, long x1)
{
  return (t->storage->data+t->storageOffset)[x0*t->stride[0]+x1*t->stride[1]];
}

TYPE &TENSOR_CLASS(Tensor)::operator()(long x0, long x1, long x2)
{
  return (t->storage->data+t->storageOffset)[x0*t->stride[0]+x1*t->stride[1]+x2*t->stride[2]];
}

TYPE &TENSOR_CLASS(Tensor)::operator()(long x0, long x1, long x2, long x3)
{
  return (t->storage->data+t->storageOffset)[x0*t->stride[0]+x1*t->stride[1]+x2*t->stride[2]+x3*t->stride[3]];
}

TENSOR_CLASS(Tensor)::~TENSOR_CLASS(Tensor)()
{
  TENSOR_FUNC(free)(t);
}


void Tprint(const TENSOR_CLASS(Tensor) *T)
{
  if(T->nDimension() == 1)
    {
      for(int y = 0 ; y < T->size(0) ; y++)
	{
	  TYPE v = TENSOR_FUNC(get1d)(T->t, y);
	  print(TYPE_FORMAT, v);
	  print(" ");
	}
      print("\n");
    }
  else if(T->nDimension() == 2)
    {
      for(int y = 0 ; y < T->size(0) ; y++)
	{
	  for(int x = 0 ; x < T->size(1) ; x++)
	    {
	      TYPE v = TENSOR_FUNC(get2d)(T->t, y, x);
	      print(TYPE_FORMAT, v);
	      print(" ");
	    }
	  print("\n");
	}
    }
  else if(T->nDimension() == 3)
    {
      for(int y = 0 ; y < T->size(0) ; y++)
	{
	  for(int x = 0 ; x < T->size(1) ; x++)
	    {
	      print(" (");
	      for(int z = 0 ; z < T->size(2) ; z++)
		{
		  TYPE v = TENSOR_FUNC(get3d)(T->t, y, x, z);
		  print(TYPE_FORMAT, v);
		  print(" ");
		}
	      print(") ");
	    }
	  print("\n");
	}
    }
}

#undef TENSOR_T_
#undef TENSOR_T
#undef TENSOR
#undef TENSOR_FUNC_TN_
#undef TENSOR_FUNC_TN
#undef TENSOR_FUNC
#undef TENSOR_CLASS_TN_
#undef TENSOR_CLASS_TN
#undef TENSOR_CLASS
