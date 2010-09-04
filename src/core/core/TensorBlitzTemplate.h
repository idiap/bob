#ifndef TORCH5SPRO_TENSOR_BLITZ_TEMPLATE_H
#define TORCH5SPRO_TENSOR_BLITZ_TEMPLATE_H

namespace Torch 
{
  /**
    * @brief The template Tensor class
    */
  template <typename T>
  class TensorTemplate: public Tensor
  {
    public:
      /**
        * Constructors
        */
      TensorTemplate();
      TensorTemplate( long dim0_);
      TensorTemplate( long dim0_, long dim1_);
      TensorTemplate( long dim0_, long dim1_, long dim2_);
      TensorTemplate( long dim0_, long dim1_, long dim2_, long dim3_);

//    Tensor(const Tensor &src);
      /**
        * Destructor
        */
      virtual ~TensorTemplate();

      /**
        * Return the number of dimensions of a Tensor
        */
      virtual int nDimension() const;
      /**
        * Return the number of elements of a particular dimension
        */
      virtual long size( int dimension_) const;
      /** 
        * Get the number of elements (over all dimensions)
        */
      virtual long sizeAll() const;
      /**
        * Tell if the Tensor is a reference or if it has his own data
        */
      virtual bool isReference() const { return m_is_reference;};

      /**
        * Set the tensor from another tensor (same type) - this will create a reference
        */
      virtual void setTensor(const TensorTemplate<T> *src_);
      virtual void setTensor( const Tensor *src) { Tensor::setTensor(src); }
      /**
        * Copy a Tensor from another one. Data will be copied, and previous content erased.
        */
      template <typename U> void copy(const TensorTemplate<U> *src_);
      virtual void copy( const Tensor *src) { Tensor::copy(src); }
      /**
        * Create a reference to a transposed Tensor
        * @param dimension1_ index of the (first) dimension to transpose
        * @param dimension2_ index of the (second) dimension to transpose
        */
      virtual void transpose(const TensorTemplate<T> *src, int dimension1_, int dimension2_);
      /**
        * Create a reference to a part of a Tensor
        * The generated Tensor will be a narrow version of the src tensor 
        * along dimension #dimension_# starting at slice #firstIndex_# and of #size_# slices
        * @param dimension_ The dimension along with only particular indices are kept
        * @param firstIndex_ The index along dimension_ where the values start to be extracted
        * @param size_ The number of indices to keep along dimension_ starting from firstIndex_
        */
      virtual void narrow(const TensorTemplate<T> *src, int dimension_, long firstIndex_, long size_);
      /**
        * Create a reference to an "hyperplane" of a Tensor
        * The generated Tensor select all the values along dimension #dimension_# at slice #sliceIndex_#
        * @param dimension_ The dimension that will be "removed"
        * @param sliceIndex The index along dimension_ from which values are selected
        */
      virtual void select(const TensorTemplate<T> *src, int dimension_, long sliceIndex_);
      /**
        * Select values from this Tensor and return a new one
        * @see select()
        */
      virtual TensorTemplate<T>* select(int dimension_, long sliceIndex_) const;
//      void unfold(const Tensor *src, int dimension_, long size_, long step_);

      /**
        * Print a Tensor
        */
      virtual void print(const char *name = NULL) const;
      virtual void sprint(const char *name, ...) const;
/*
      // Access to the raw data
      /void* dataW();
      const void* dataR() const;

      // Get the size of an element
      int typeSize() const;
*/
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

      /**
        * Set a value of a Tensor
        */
  		virtual void set(long, char);
      virtual void set(long, long, char);
      virtual void set(long, long, long, char);
      virtual void set(long, long, long, long, char);

      virtual void set(long, short);
      virtual void set(long, long, short);
      virtual void set(long, long, long, short);
      virtual void set(long, long, long, long, short);

      virtual void set(long, int);
      virtual void set(long, long, int);
      virtual void set(long, long, long, int);
      virtual void set(long, long, long, long, int);

      virtual void set(long, long);
      virtual void set(long, long, long);
      virtual void set(long, long, long, long);
      virtual void set(long, long, long, long, long);

      virtual void set(long, float);
      virtual void set(long, long, float);
      virtual void set(long, long, long, float);
      virtual void set(long, long, long, long, float);

      virtual void set(long, double);
      virtual void set(long, long, double);
      virtual void set(long, long, long, double);
      virtual void set(long, long, long, long, double);

      /**
        * Return the sum of the elements of a Tensor
        */
      T sum() const;

      /**
        * Get a value of a Tensor
        */
      T get(long) const;
      T get(long, long) const;
      T get(long, long, long) const;
      T get(long, long, long, long) const;

      T &operator()(long);
      T &operator()(long, long);
      T &operator()(long, long, long);
      T &operator()(long, long, long, long);

      const T &operator()(long) const;
      const T &operator()(long, long) const;
      const T &operator()(long, long, long) const;
      const T &operator()(long, long, long, long) const;

      const T*  dataR() const;
      T*        dataW();
      long      stride(int dim) const;
      void      resetFromData();


    private:
      typedef typename blitz::Array<T,1>                        Tensor_1D_Type;
      typedef typename blitz::Array<T,2>                        Tensor_2D_Type;
      typedef typename blitz::Array<T,3>                        Tensor_3D_Type;
      typedef typename blitz::Array<T,4>                        Tensor_4D_Type;


      /** 
        * Get the multi_array pointer where data are stored
        */
      Tensor_1D_Type* getArray1D()  const        { return m_array_1D; }
      Tensor_2D_Type* getArray2D()  const        { return m_array_2D; }
      Tensor_3D_Type* getArray3D()  const        { return m_array_3D; }
      Tensor_4D_Type* getArray4D()  const        { return m_array_4D; }

      /** 
        * Get the multi_array view from which data of the Tensor are taken
        */
      Tensor_1D_Type*   getView1D()   const        { return m_view_1D; }
      Tensor_2D_Type*   getView2D()   const        { return m_view_2D; }
      Tensor_3D_Type*   getView3D()   const        { return m_view_3D; }
      Tensor_4D_Type*   getView4D()   const        { return m_view_4D; }
  
      /**
        * Initialize the indices of the dimensions of a Tensor wrt to the multi_array view
        * The corresponding array of indices may be affected by the transpose function
        * @see transpose()
        */
      void            initDimensionIndex();
      /**
        * Set the array of indices
        * @see initDimensionIndex()
        */
      bool            setDimensionIndex(const size_t dimension[4]);
      /**
        * Remove a dimension in the array of indices
        * This occurs when performing a select() for instance
        * @see initDimensionIndex()
        * @see select()
        */
      bool            removeDimensionIndex(const size_t dimension);
      /**
        * Get the dimension of the multi_array view used by this Tensor instance
        */
      const size_t*   getDimensionIndex() const { return m_dimension_index; }
      /**
        * Cleanup the dynamical members of a Tensor
        */
      void            cleanup();
      /**
        * Get the dimension of the multi_array used by this Tensor instance
        */
      size_t          getDimensionArray() const { return m_n_dimensions_array; }

      /**
        * Return the indices in the order required by the multi_array view
        */
      void            indicesCorrectOrder( long dim0, long dim1, long* rdim) const;
      void            indicesCorrectOrder( long dim0, long dim1, long dim2, long *rdim) const;
      void            indicesCorrectOrder( long dim0, long dim1, long dim2, long dim3, long *rdim) const;

      /**
        * Set the datatype of a Tensor
        * This function has specializations and call the parent function setDataTypeMain()
        * @see Tensor::setDataTypeMain()
        */
      virtual void setDataType();

      void computeDataRW();
      void computeStride();
      void Tprint() const;

      size_t m_n_dimensions;
      size_t m_n_dimensions_array;
      bool m_is_reference;

      Tensor_1D_Type  *m_array_1D;
      Tensor_2D_Type  *m_array_2D;
      Tensor_3D_Type  *m_array_3D;
      Tensor_4D_Type  *m_array_4D;

      Tensor_1D_Type  *m_view_1D;
      Tensor_2D_Type  *m_view_2D;
      Tensor_3D_Type  *m_view_3D;
      Tensor_4D_Type  *m_view_4D;

      T               *m_data_RW;
      long            m_stride[4];


      size_t m_dimension_index[4];
  };    
}

#endif
