/**
  * @file TensorBlitzTemplateAA.h
  * @author <a href="mailto:andre.anjos@idiap.ch">Andre Anjos</a> 
  * @author <a href="mailto:Laurent.El-Shafey@idiap.ch">Laurent El Shafey</a> 
  *
  * @brief Wrapper using blitz::Array
  */

#ifndef TORCH5SPRO_TENSOR_BLITZ_AA_TEMPLATE_H
#define TORCH5SPRO_TENSOR_BLITZ_AA_TEMPLATE_H

#include <cstdarg>

namespace Torch 
{
  /**
   * @brief The template Tensor class
   */
  template <typename T> class TensorTemplate: public Tensor
  {
    public:
      /**
       * Constructors
       */
      TensorTemplate();
      TensorTemplate(long dim0_);
      TensorTemplate(long dim0_, long dim1_);
      TensorTemplate(long dim0_, long dim1_, long dim2_);
      TensorTemplate(long dim0_, long dim1_, long dim2_, long dim3_);

      /**
       * Destructor
       */
      virtual ~TensorTemplate();

      /**
       * Copy constructor and operator
       */
      TensorTemplate(const TensorTemplate<T>& other):
        m_n_dimensions(0), 
        m_array_1D(0), 
        m_array_2D(0), 
        m_array_3D(0), 
        m_array_4D(0),
        m_data_RW(0)
      { 
        copy(&other); 
      }

      Tensor& operator=(const TensorTemplate<T>& other) { 
        copy(&other); 
        return *this;
      }

      /**
       * Return the number of dimensions of a Tensor
       */
      virtual inline int nDimension() const { return m_n_dimensions; }

      /**
       * Return the number of elements of a particular dimension
       */
      virtual long size(int dimension_) const;

      /** 
       * Get the number of elements (over all dimensions)
       */
      virtual long sizeAll() const;

      /**
       * Tell if the Tensor is a reference or if it has his own data
       */
      virtual inline bool isReference() const { return true; };

      /**
       * Set the tensor from another tensor (same type) - this will create a
       * reference
       */
      virtual void setTensor(const TensorTemplate<T> *src_);
      virtual void setTensor(const Tensor *src) { Tensor::setTensor(src); }

      /**
       * Copy a Tensor from another one. Data will be copied, and previous
       * content erased.
       */
      template <typename U> void copy(const TensorTemplate<U> *src_);
      virtual inline void copy(const Tensor *src) { Tensor::copy(src); }

      /**
       * Create a reference to a transposed Tensor
       * @param dimension1_ index of the (first) dimension to transpose
       * @param dimension2_ index of the (second) dimension to transpose
       */
      virtual void transpose(const TensorTemplate<T> *src, int dimension1_,
          int dimension2_);
      /**
       * Create a reference to a part of a Tensor
       * The generated Tensor will be a narrow version of the src tensor 
       * along dimension #dimension_# starting at slice #firstIndex_# and of
       * #size_# slices
       * @param dimension_ The dimension along with only particular indices are
       * kept
       * @param firstIndex_ The index along dimension_ where the values start
       * to be extracted
       * @param size_ The number of indices to keep along dimension_ starting
       * from firstIndex_
       */
      virtual void narrow(const TensorTemplate<T> *src, int dimension_,
          long firstIndex_, long size_);

      /**
       * Create a reference to an "hyperplane" of a Tensor
       * The generated Tensor select all the values along dimension
       * #dimension_# at slice #sliceIndex_#
       * @param dimension_ The dimension that will be "removed"
       * @param sliceIndex The index along dimension_ from which values are
       * selected
       */
      virtual void select(const TensorTemplate<T> *src, int dimension_, 
          long sliceIndex_);

      /**
       * Select values from this Tensor and return a new one
       * @see select()
       */
      virtual TensorTemplate<T>* select(int dimension_, 
          long sliceIndex_) const;

      /**
       * Print a Tensor
       */
      virtual void print(const char *name = NULL) const;
      virtual void sprint(const char *name, ...) const;

      /**
       * Fill in a Tensor with a particular value
       * @param value The value which will be set
       */
      virtual void fill(T value);

      /**
       * Resize a Tensor 
       */
      template <typename U> void resizeAs(const TensorTemplate<U> &src_);
      virtual void resize(long);
      virtual void resize(long, long);
      virtual void resize(long, long, long);
      virtual void resize(long, long, long, long);

      virtual inline void set(long d0, char v) { tset(d0, v); }
      virtual inline void set(long d0, long d1, char v) { tset(d0, d1, v); }
      virtual inline void set(long d0, long d1, long d2, char v)
      { tset(d0, d1, d2, v); }
      virtual inline void set(long d0, long d1, long d2, long d3, char v)
      { tset(d0, d1, d2, d3, v); }

      virtual inline void set(long d0, short v) { tset(d0, v); }
      virtual inline void set(long d0, long d1, short v) { tset(d0, d1, v); }
      virtual inline void set(long d0, long d1, long d2, short v)
      { tset(d0, d1, d2, v); }
      virtual inline void set(long d0, long d1, long d2, long d3, short v)
      { tset(d0, d1, d2, d3, v); }
      
      virtual inline void set(long d0, int v) { tset(d0, v); }
      virtual inline void set(long d0, long d1, int v) { tset(d0, d1, v); }
      virtual inline void set(long d0, long d1, long d2, int v)
      { tset(d0, d1, d2, v); }
      virtual inline void set(long d0, long d1, long d2, long d3, int v)
      { tset(d0, d1, d2, d3, v); }
      
      virtual inline void set(long d0, long v) { tset(d0, v); }
      virtual inline void set(long d0, long d1, long v) { tset(d0, d1, v); }
      virtual inline void set(long d0, long d1, long d2, long v)
      { tset(d0, d1, d2, v); }
      virtual inline void set(long d0, long d1, long d2, long d3, long v)
      { tset(d0, d1, d2, d3, v); }
      
      virtual inline void set(long d0, float v) { tset(d0, v); }
      virtual inline void set(long d0, long d1, float v) { tset(d0, d1, v); }
      virtual inline void set(long d0, long d1, long d2, float v)
      { tset(d0, d1, d2, v); }
      virtual inline void set(long d0, long d1, long d2, long d3, float v)
      { tset(d0, d1, d2, d3, v); }
      
      virtual inline void set(long d0, double v) { tset(d0, v); }
      virtual inline void set(long d0, long d1, double v) { tset(d0, d1, v); }
      virtual inline void set(long d0, long d1, long d2, double v)
      { tset(d0, d1, d2, v); }
      virtual inline void set(long d0, long d1, long d2, long d3, double v)
      { tset(d0, d1, d2, d3, v); }

      /**
       * Return the sum of the elements of a Tensor
       */
      T sum() const;

      /**
       * Get a value of a Tensor
       */
      T& operator()(long d0);
      inline T& operator()(long d0, long d1)
      { return (*m_array_2D)((int)d0, (int)d1); }
      inline T& operator()(long d0, long d1, long d2)
      { return (*m_array_3D)((int)d0, (int)d1, (int)d2); }
      inline T& operator()(long d0, long d1, long d2, long d3)
      { return (*m_array_4D)((int)d0, (int)d1, (int)d2, (int)d3); }

      const T& operator()(long d0) const;
      inline const T& operator()(long d0, long d1) const 
      { return (*m_array_2D)((int)d0, (int)d1); }
      inline const T& operator()(long d0, long d1, long d2) const
      { return (*m_array_3D)((int)d0, (int)d1, (int)d2); }
      inline const T& operator()(long d0, long d1, long d2, long d3) const
      { return (*m_array_4D)((int)d0, (int)d1, (int)d2, (int)d3); }

      inline T get(long d0) const { return (*this)(d0); }
      inline T get(long d0, long d1) const { return (*this)(d0, d1); }
      inline T get(long d0, long d1, long d2) const 
      { return (*this)(d0, d1, d2); }
      inline T get(long d0, long d1, long d2, long d3) const
      { return (*this)(d0, d1, d2, d3); }

      const T* dataR() const;
      T* dataW();
      long stride(int dim) const;
      void resetFromData();

    private:

      /**
       * Set a value of a Tensor
       */
      template<typename U> inline void tset(long d0, U v) {
        (*m_array_1D)((int)d0) = (T)v; 
      }

      template<typename U> inline void tset(long d0, long d1, U v) {
        (*m_array_2D)((int)d0, (int)d1) = (T)v; 
      }

      template<typename U> inline void tset(long d0, long d1, long d2, U v) {
        (*m_array_3D)((int)d0, (int)d1, (int)d2) = (T)v; 
      }

      template<typename U> inline void tset(long d0, long d1, long d2, long d3,
          U v) {
        (*m_array_4D)((int)d0, (int)d1, (int)d2, (int)d3) = (T)v; 
      }

      /**
       * Cleans-up tensor storage
       */
      void cleanup();

      /**
       * Set the datatype of a Tensor
       * This function has specializations and call the parent function
       * Torch::Tensor::setDataTypeMain()
       */
      virtual void setDataType();

      void computeDataRW();
      void computeStride() const;
      void Tprint() const;

    private: //representation

      size_t m_n_dimensions;
      blitz::Array<T,1> *m_array_1D;
      blitz::Array<T,2> *m_array_2D;
      blitz::Array<T,3> *m_array_3D;
      blitz::Array<T,4> *m_array_4D;

      T* m_data_RW;
      size_t m_stride[4];
  };    

  template <typename T> void TensorTemplate<T>::setDataType() { 
    setDataTypeMain(Tensor::Undefined); 
  }

  template <> inline void TensorTemplate<char>::setDataType() { 
    setDataTypeMain(Tensor::Char); 
  }

  template <> inline void TensorTemplate<short>::setDataType() { 
    setDataTypeMain(Tensor::Short); 
  }

  template <> inline void TensorTemplate<int>::setDataType() { 
    setDataTypeMain(Tensor::Int); 
  }

  template <> inline void TensorTemplate<long>::setDataType() { 
    setDataTypeMain(Tensor::Long); 
  }

  template <> inline void TensorTemplate<float>::setDataType() { 
    setDataTypeMain(Tensor::Float); 
  }

  template <> inline void TensorTemplate<double>::setDataType() { 
    setDataTypeMain(Tensor::Double); 
  }

}

template <typename T> Torch::TensorTemplate<T>::TensorTemplate(): 
  m_n_dimensions(0), 
  m_array_1D(0), 
  m_array_2D(0), 
  m_array_3D(0), 
  m_array_4D(0),
  m_data_RW(0)
{
  setDataType();
}

template <typename T> Torch::TensorTemplate<T>::TensorTemplate(long dim0): 
  m_n_dimensions(1), 
  m_array_1D(0), 
  m_array_2D(0), 
  m_array_3D(0), 
  m_array_4D(0),
  m_data_RW(0)
{
  setDataType();
  m_array_1D = new blitz::Array<T,1>((int)dim0);
}

template <typename T> Torch::TensorTemplate<T>::TensorTemplate(long dim0,
    long dim1): 
  m_n_dimensions(2), 
  m_array_1D(0), 
  m_array_2D(0), 
  m_array_3D(0), 
  m_array_4D(0),
  m_data_RW(0)
{
  setDataType();
  m_array_2D = new blitz::Array<T,2>((int)dim0, (int)dim1);
}

template <typename T> Torch::TensorTemplate<T>::TensorTemplate(long dim0,
    long dim1, long dim2): 
  m_n_dimensions(3), 
  m_array_1D(0), 
  m_array_2D(0), 
  m_array_3D(0), 
  m_array_4D(0),
  m_data_RW(0)
{
  setDataType();
  m_array_3D = new blitz::Array<T,3>((int)dim0, (int)dim1, (int)dim2);
}

template <typename T> Torch::TensorTemplate<T>::TensorTemplate(long dim0, 
    long dim1, long dim2, long dim3): 
  m_n_dimensions(4), 
  m_array_1D(0), 
  m_array_2D(0), 
  m_array_3D(0), 
  m_array_4D(0),
  m_data_RW(0)
{
  setDataType();
  m_array_4D = new blitz::Array<T,4>((int)dim0, (int)dim1, (int)dim2, (int)dim3);
}

template <typename T> void Torch::TensorTemplate<T>::cleanup() {
  delete m_array_1D;
  m_array_1D = 0;
  delete m_array_2D;
  m_array_2D = 0;
  delete m_array_3D;
  m_array_3D = 0;
  delete m_array_4D;
  m_array_4D = 0;
  delete[] m_data_RW;
  m_data_RW = 0;
  m_n_dimensions = 0;
}

template <typename T> Torch::TensorTemplate<T>::~TensorTemplate() {
  cleanup();
}

template <typename T> long
Torch::TensorTemplate<T>::size(int dimension) const {
  switch(m_n_dimensions) {
    case 1: return m_array_1D->extent(dimension);
    case 2: return m_array_2D->extent(dimension);
    case 3: return m_array_3D->extent(dimension);
    case 4: return m_array_4D->extent(dimension);
    default: return 0;
  }
}

template <typename T> long Torch::TensorTemplate<T>::sizeAll() const {
  switch(m_n_dimensions) {
    case 1: return m_array_1D->size();
    case 2: return m_array_2D->size();
    case 3: return m_array_3D->size();
    case 4: return m_array_4D->size();
    default: return 0;
  }
}

template <typename T> void 
Torch::TensorTemplate<T>::setTensor(const Torch::TensorTemplate<T> *src) {
  cleanup();
  m_n_dimensions = src->nDimension();
  switch(m_n_dimensions) {
    case 1: 
      m_array_1D = new blitz::Array<T,1>(*src->m_array_1D);
      break;
    case 2:
      m_array_2D = new blitz::Array<T,2>(*src->m_array_2D);
      break;
    case 3:
      m_array_3D = new blitz::Array<T,3>(*src->m_array_3D);
      break;
    case 4:
      m_array_4D = new blitz::Array<T,4>(*src->m_array_4D);
      break;
  }
}

template <typename T> template <typename U> void 
Torch::TensorTemplate<T>::copy(const Torch::TensorTemplate<U> *src) {
  cleanup();
  m_n_dimensions = src->nDimension();
  switch(m_n_dimensions) {
    case 1: 
      m_array_1D = new blitz::Array<T,1>(src->size(0));
      for (int i=0; i<src->size(0); ++i) this->tset(i, (*src)(i));
      break;
    case 2:
      m_array_2D = new blitz::Array<T,2>(src->size(0), src->size(1));
      for (int i=0; i<src->size(0); ++i) 
        for (int j=0; j<src->size(1); ++j) this->tset(i, j, (*src)(i, j));
      break;
    case 3:
      m_array_3D = new blitz::Array<T,3>(src->size(0), src->size(1), src->size(2));
      for (int i=0; i<src->size(0); ++i) 
        for (int j=0; j<src->size(1); ++j) 
          for (int k=0; k<src->size(2); ++k) this->tset(i, j, k, (*src)(i, j, k));
      break;
    case 4:
      m_array_4D = new blitz::Array<T,4>(src->size(0), src->size(1), src->size(2), src->size(3));
      for (int i=0; i<src->size(0); ++i) 
        for (int j=0; j<src->size(1); ++j) 
          for (int k=0; k<src->size(2); ++k)
            for (int l=0; l<src->size(3); ++l) this->tset(i, j, k, l, (*src)(i, j, k, l));
      break;
  }
}

template <typename T> void Torch::TensorTemplate<T>::transpose
(const Torch::TensorTemplate<T> *src, int dimension1, int dimension2) {
  setTensor(src);
  switch(m_n_dimensions) {
    case 2:
      m_array_2D->transposeSelf(1, 0);
      break;
    case 3:
      {
        int tdim[3] = {0, 1, 2};
        tdim[dimension1] = dimension2;
        tdim[dimension2] = dimension1;
        m_array_3D->transposeSelf(tdim[0], tdim[1], tdim[2]);
      }
      break;
    case 4:
      {
        int tdim[4] = {0, 1, 2, 3};
        tdim[dimension1] = dimension2;
        tdim[dimension2] = dimension1;
        m_array_4D->transposeSelf(tdim[0], tdim[1], tdim[2], tdim[3]);
      }
      break;
  }
}

template <typename T> void Torch::TensorTemplate<T>::print (const char *name)
  const {
  if(name != NULL) std::cout << "Tensor " << name << ":" << std::endl;
  Tprint();
}

template <typename T> void Torch::TensorTemplate<T>::sprint 
(const char *name, ...) const {
  if(name != NULL) {
    char _msg[512];
    va_list args;
    va_start(args, name);
    vsprintf(_msg, name, args);
    std::cout << "Tensor " << _msg << ":" << std::endl;
    fflush(stdout);
    va_end(args);
  }
  Tprint();
}

template <typename T> void Torch::TensorTemplate<T>::Tprint() const {
  switch(m_n_dimensions) {
    case 0:
      std::cout << "Oops !! I can't print a 0D tensor :-(" << std::endl;
      break;
    case 1:
      std::cout << *m_array_1D << std::endl;
      break;
    case 2:
      std::cout << *m_array_2D << std::endl;
      break;
    case 3:
      std::cout << *m_array_3D << std::endl;
      break;
    case 4:
      std::cout << *m_array_4D << std::endl;
      break;
  }
}

template <typename T> template <typename U> void
Torch::TensorTemplate<T>::resizeAs(const Torch::TensorTemplate<U> &src) {
  cleanup();
  m_n_dimensions = src.nDimension();
  switch(m_n_dimensions) {
    case 1: 
      m_array_1D = new blitz::Array<T,1>((int)src.size(0));
      break;
    case 2: 
      m_array_2D = new blitz::Array<T,2>((int)src.size(0), (int)src.size(1));
      break;
    case 3:
      m_array_3D = new blitz::Array<T,3>((int)src.size(0), (int)src.size(1),
          (int)src.size(2));
      break;
    case 4:
      m_array_4D = new blitz::Array<T,4>((int)src.size(0), (int)src.size(1),
          (int)src.size(2), (int)src.size(3));
      break;
  }
}

template <typename T> void Torch::TensorTemplate<T>::resize(long size0) {
  cleanup();
  m_n_dimensions = 1;
  m_array_1D = new blitz::Array<T,1>((int)size0);
}

template <typename T> void
Torch::TensorTemplate<T>::resize(long size0, long size1) {
  cleanup();
  m_n_dimensions = 2;
  m_array_2D = new blitz::Array<T,2>((int)size0, (int)size1);
}

template <typename T> void Torch::TensorTemplate<T>::resize
(long size0, long size1, long size2) {
  cleanup();
  m_n_dimensions = 3;
  m_array_3D = new blitz::Array<T,3>((int)size0, (int)size1, (int)size2);
}

template <typename T> void Torch::TensorTemplate<T>::resize
(long size0, long size1, long size2, long size3) {
  cleanup();
  m_n_dimensions = 4;
  m_array_4D = new blitz::Array<T,4>((int)size0, (int)size1, (int)size2, 
      (int)size3);
}

template <typename T> T Torch::TensorTemplate<T>::sum() const {
  switch(m_n_dimensions) {
    case 1:
      return blitz::sum(*m_array_1D);
    case 2:
      return blitz::sum(*m_array_2D);
    case 3:
      return blitz::sum(*m_array_3D);
    case 4:
      return blitz::sum(*m_array_4D);
    default:
      break;
  }
  return 0;
}

template <typename T> T& Torch::TensorTemplate<T>::operator()(long dim0) {
  switch(m_n_dimensions) {
    case 1:
      return (*m_array_1D)((int)dim0);
    case 2:
      {
        long ndim0=dim0 % size(0);
        long ndim1=dim0 / size(0);
        return (*this)(ndim0, ndim1);
      }
    case 3:
      {
        long ndim2=dim0 / (size(0)*size(1));
        dim0 -= (size(0)*size(1))*ndim2;
        long ndim1=dim0 / size(0);
        long ndim0=dim0 % size(0);
        return (*this)(ndim0, ndim1, ndim2);
      }
    case 4:
      {
        long ndim3=dim0 / (size(0)*size(1)*size(2));
        dim0 -= (size(0)*size(1)*size(2))*ndim3;
        long ndim2=dim0 / (size(0)*size(1));
        dim0 -= (size(0)*size(1))*ndim2;
        long ndim1=dim0 / size(0);
        long ndim0=dim0 % size(0);
        return (*this)(ndim0, ndim1, ndim2, ndim3);
      }
    default:
      std::cerr << "Error Tensor::(): do not know what to do with " << m_n_dimensions << " dimensions." << std::endl;
      std::exit(-1);
  }
}

template <typename T> const T& Torch::TensorTemplate<T>::operator()(long dim0) const {
  long ndim0;
  long ndim1;
  long ndim2;
  long ndim3;
  switch(m_n_dimensions) {
    case 1:
      return (*m_array_1D)((int)dim0);
    case 2:
      ndim0=dim0 % size(0);
      ndim1=dim0 / size(0);
      return (*this)(ndim0, ndim1);
    case 3:
      ndim2=dim0 / (size(0)*size(1));
      dim0 -= (size(0)*size(1))*ndim2;
      ndim1=dim0 / size(0);
      ndim0=dim0 % size(0);
      return (*this)(ndim0, ndim1, ndim2);
    case 4:
      ndim3=dim0 / (size(0)*size(1)*size(2));
      dim0 -= (size(0)*size(1)*size(2))*ndim3;
      ndim2=dim0 / (size(0)*size(1));
      dim0 -= (size(0)*size(1))*ndim2;
      ndim1=dim0 / size(0);
      ndim0=dim0 % size(0);
      return (*this)(ndim0, ndim1, ndim2, ndim3);
    default:
      std::cerr << "Error Tensor::(): do not know what to do with " << m_n_dimensions << " dimensions." << std::endl;
      std::exit(-1);
  }
}

template <typename T> void Torch::TensorTemplate<T>::computeDataRW() {
  switch(m_n_dimensions)
  {
    case 1:
      for(int i = 0; i<size(0); i++)
        m_data_RW[i] = (*m_array_1D)(i);
      break;
    case 2:
      for(int j = 0; j<size(1); j++)
        for(int i = 0; i<size(0); i++)
          m_data_RW[i+j*m_stride[1]] = (*m_array_2D)(i, j);
      break;
    case 3:
      for(int k = 0; k<size(2); k++)
        for(int j = 0; j<size(1); j++)
          for(int i = 0; i<size(0); i++)
            m_data_RW[i+j*m_stride[1]+k*m_stride[2]] = (*m_array_3D)(i, j, k);
      break;
    case 4:
      for(int l = 0; l<size(3); l++)
        for(int k = 0; k<size(2); k++)
          for(int j = 0; j<size(1); j++)
            for(int i = 0; i<size(0); i++)
              m_data_RW[i+j*m_stride[1]+k*m_stride[2]+l*m_stride[3]] = (*m_array_4D)(i, j, k, l);
    default:
      break;
  }
}

template <typename T> const T* Torch::TensorTemplate<T>::dataR() const {
  Torch::TensorTemplate<T>* self = const_cast<Torch::TensorTemplate<T>*>(this);
  size_t array_length = sizeAll();
  if (self->m_data_RW) delete[] self->m_data_RW;
  self->m_data_RW = new T[array_length];
  computeStride();
  self->computeDataRW();
  return self->m_data_RW;
}

template <typename T> T* Torch::TensorTemplate<T>::dataW() {
  size_t array_length = sizeAll();
  if(m_data_RW) delete[] m_data_RW;
  m_data_RW = new T[array_length];
  computeStride();
  computeDataRW();
  return m_data_RW;
}

template <typename T> void Torch::TensorTemplate<T>::computeStride() const {
  Torch::TensorTemplate<T>* self = const_cast<Torch::TensorTemplate<T>*>(this);
  switch(m_n_dimensions) {
    case 1:
      self->m_stride[0] = 1;
      self->m_stride[1] = self->m_stride[2] = self->m_stride[3] = 0;
      break;
    case 2:
      self->m_stride[0] = 1;
      self->m_stride[1] = size(0);
      self->m_stride[2] = self->m_stride[3] = 0;
      break;
    case 3:
      self->m_stride[0] = 1;
      self->m_stride[1] = size(0);
      self->m_stride[2] = size(0)*size(1);
      self->m_stride[3] = 0;
      break;
    case 4:
      self->m_stride[0] = 1;
      self->m_stride[1] = size(0);
      self->m_stride[2] = size(0)*size(1);
      self->m_stride[3] = size(0)*size(1)*size(2);
    default:
      break;
  }
}

template <typename T> long Torch::TensorTemplate<T>::stride(int dim) const {
  computeStride();
  return m_stride[(size_t)dim];
}

template <typename T> void Torch::TensorTemplate<T>::resetFromData() {
  switch(m_n_dimensions) {
    case 1:
      for(int i = 0; i<size(0); ++i)
        (*m_array_1D)(i) = m_data_RW[i];
      break;
    case 2:
      for(int j = 0; j<size(1); ++j)
        for(int i = 0; i<size(0); ++i)
          (*m_array_2D)(i, j) = m_data_RW[i+j*m_stride[1]];
      break;
    case 3:
      for(int k = 0; k<size(2); ++k)
        for(int j = 0; j<size(1); ++j)
          for(int i = 0; i<size(0); ++i)
            (*m_array_3D)(i, j, k) = m_data_RW[i+j*m_stride[1]+k*m_stride[2]];
      break;
    case 4:
      for(int l = 0; l<size(3); ++l)
        for(int k = 0; k<size(2); ++k)
          for(int j = 0; j<size(1); ++j)
            for(int i = 0; i<size(0); ++i)
              (*m_array_4D)(i, j, k, l) = m_data_RW[i+j*m_stride[1]+k*m_stride[2]+l*m_stride[3]];
    default:
      break;
  }
}

template <typename T> void Torch::TensorTemplate<T>::narrow
(const Torch::TensorTemplate<T> *src, int dimension, long firstIndex, 
 long size) {
  cleanup();
  blitz::Range all_range = blitz::Range::all();
  blitz::Range narrowed(firstIndex, firstIndex+size-1);
  switch(src->m_n_dimensions) {
    case 1:
      m_n_dimensions = 1;
      switch(dimension) {
        case 0:
          m_array_1D = new blitz::Array<T,1>((*src->m_array_1D)(narrowed));
          break;
      }
      break;
    case 2: 
      m_n_dimensions = 2;
      switch(dimension) {
        case 0:
          m_array_2D = new blitz::Array<T,2>((*src->m_array_2D)(narrowed, all_range));
          break;
        case 1:
          m_array_2D = new blitz::Array<T,2>((*src->m_array_2D)(all_range, narrowed));
          break;
      }
      break;
    case 3:
      m_n_dimensions = 3;
      switch(dimension) {
        case 0:
          m_array_3D = new blitz::Array<T,3>((*src->m_array_3D)(narrowed, all_range, all_range));
          break;
        case 1:
          m_array_3D = new blitz::Array<T,3>((*src->m_array_3D)(all_range, narrowed, all_range));
          break;
        case 2:
          m_array_3D = new blitz::Array<T,3>((*src->m_array_3D)(all_range, all_range, narrowed));
          break;
      }
      break;
    case 4:
      m_n_dimensions = 4;
      switch(dimension) {
        case 0:
          m_array_4D = new blitz::Array<T,4>((*src->m_array_4D)(narrowed, all_range, all_range, all_range));
          break;
        case 1:
          m_array_4D = new blitz::Array<T,4>((*src->m_array_4D)(all_range, narrowed, all_range, all_range));
          break;
        case 2:
          m_array_4D = new blitz::Array<T,4>((*src->m_array_4D)(all_range, all_range, narrowed, all_range));
          break;
        case 3:
          m_array_4D = new blitz::Array<T,4>((*src->m_array_4D)(all_range, all_range, all_range, narrowed));
          break;
      }
      break;
  }
}

template <typename T> void Torch::TensorTemplate<T>::select
(const Torch::TensorTemplate<T> *src, int dimension, long sliceIndex) {
  cleanup();
  blitz::Range all_range = blitz::Range::all();
  switch(src->m_n_dimensions) {
    case 1:
      std::cerr << "Tensor::select(): Cannot select on a 1D tensor" 
                << std::endl;
      break;
    case 2: 
      switch(dimension) {
        case 0:
          m_array_1D = new blitz::Array<T,1>((*src->m_array_2D)(sliceIndex, all_range));
          m_n_dimensions = 1;
          break;
        case 1:
          m_array_1D = new blitz::Array<T,1>((*src->m_array_2D)(all_range, sliceIndex));
          m_n_dimensions = 1;
          break;
      }
      break;
    case 3:
      switch(dimension) {
        case 0:
          m_array_2D = new blitz::Array<T,2>((*src->m_array_3D)(sliceIndex, all_range, all_range));
          m_n_dimensions = 2;
          break;
        case 1:
          m_array_2D = new blitz::Array<T,2>((*src->m_array_3D)(all_range, sliceIndex, all_range));
          m_n_dimensions = 2;
          break;
        case 2:
          m_array_2D = new blitz::Array<T,2>((*src->m_array_3D)(all_range, all_range, sliceIndex));
          m_n_dimensions = 2;
          break;
      }
      break;
    case 4:
      switch(dimension) {
        case 0:
          m_array_3D = new blitz::Array<T,3>((*src->m_array_4D)(sliceIndex, all_range, all_range, all_range));
          m_n_dimensions = 3;
          break;
        case 1:
          m_array_3D = new blitz::Array<T,3>((*src->m_array_4D)(all_range, sliceIndex, all_range, all_range));
          m_n_dimensions = 3;
          break;
        case 2:
          m_array_3D = new blitz::Array<T,3>((*src->m_array_4D)(all_range, all_range, sliceIndex, all_range));
          m_n_dimensions = 3;
          break;
        case 3:
          m_array_3D = new blitz::Array<T,3>((*src->m_array_4D)(all_range, all_range, all_range, sliceIndex));
          m_n_dimensions = 3;
          break;
      }
      break;
  }
}

template <typename T> Torch::TensorTemplate<T>* 
Torch::TensorTemplate<T>::select(int dimension, long sliceIndex) const {
  Torch::TensorTemplate<T>* result = new Torch::TensorTemplate<T>();
  result->select(this, dimension, sliceIndex);
  return result;
}

template <typename T> void Torch::TensorTemplate<T>::fill(T val) {
  switch(m_n_dimensions) {
    case 1:
      (*m_array_1D) = val;
      return;
    case 2:
      (*m_array_2D) = val;
      return;
    case 3:
      (*m_array_3D) = val;
      return;
    case 4:
      (*m_array_4D) = val;
      return;
  }
}

#endif /* TORCH5SPRO_TENSOR_BLITZ_AA_TEMPLATE_H */
