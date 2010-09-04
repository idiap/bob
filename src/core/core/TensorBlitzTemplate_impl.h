#ifndef TORCH5SPRO_TENSOR_BLITZ_TEMPLATE_IMPL_H
#define TORCH5SPRO_TENSOR_BLITZ_TEMPLATE_IMPL_H

namespace Torch 
{
/*
  template <typename T> Tensor<T>::Tensor(const Tensor &src): nDimension(0)
  {
    reinit( src.storage, src.storageOffset, src.nDimension, src.size, src.stride);
  }
*/

  template <typename T> TensorTemplate<T>::TensorTemplate(): 
    m_n_dimensions(0), 
    m_n_dimensions_array(0),
    m_is_reference(false),
    m_array_1D(0), m_array_2D(0), m_array_3D(0), m_array_4D(0),
    m_view_1D(0), m_view_2D(0), m_view_3D(0), m_view_4D(0),
    m_data_RW(0)
  {
    initDimensionIndex();
    setDataType();
  }

  template <typename T> TensorTemplate<T>::TensorTemplate(long dim0): 
    m_n_dimensions(1), 
    m_n_dimensions_array(1),
    m_is_reference(false),
    m_array_2D(0), m_array_3D(0), m_array_4D(0),
    m_view_2D(0), m_view_3D(0), m_view_4D(0),
    m_data_RW(0)
  {
    initDimensionIndex();
    setDataType();
    m_array_1D = new Tensor_1D_Type(dim0);
    m_view_1D = new Tensor_1D_Type( (*m_array_1D)(blitz::Range::all()) );
  }

  template <typename T> TensorTemplate<T>::TensorTemplate(long dim0, long dim1): 
    m_n_dimensions(2), 
    m_n_dimensions_array(2),
    m_is_reference(false),  
    m_array_1D(0), m_array_3D(0), m_array_4D(0),
    m_view_1D(0), m_view_3D(0), m_view_4D(0),
    m_data_RW(0)
  {
    initDimensionIndex();
    setDataType();
    m_array_2D = new Tensor_2D_Type(dim0, dim1);
    m_view_2D = new Tensor_2D_Type((*m_array_2D)( blitz::Range::all(), blitz::Range::all()));
  }

  template <typename T> TensorTemplate<T>::TensorTemplate(long dim0, long dim1, long dim2): 
    m_n_dimensions(3), 
    m_n_dimensions_array(3),
    m_is_reference(false),
    m_array_1D(0), m_array_2D(0), m_array_4D(0),
    m_view_1D(0), m_view_2D(0), m_view_4D(0),
    m_data_RW(0)
  {
    initDimensionIndex();
    setDataType();
    m_array_3D = new Tensor_3D_Type(dim0, dim1, dim2);
    m_view_3D = new Tensor_3D_Type((*m_array_3D)( blitz::Range::all(), blitz::Range::all(), blitz::Range::all() ));
  }

  template <typename T> TensorTemplate<T>::TensorTemplate(long dim0, long dim1, long dim2, long dim3): 
    m_n_dimensions(4), 
    m_n_dimensions_array(4),
    m_is_reference(false),
    m_array_1D(0), m_array_2D(0), m_array_3D(0),
    m_view_1D(0), m_view_2D(0), m_view_3D(0),
    m_data_RW(0)
  {
    initDimensionIndex();
    setDataType();
    m_array_4D = new Tensor_4D_Type(dim0, dim1, dim2, dim3);
    m_view_4D = new Tensor_4D_Type((*m_array_4D)( blitz::Range::all(), blitz::Range::all(), blitz::Range::all(), blitz::Range::all()));
  }


  template <typename T> void
  TensorTemplate<T>::cleanup()
  {
    // Arrays should only be deleted in case the Tensor is not a reference
    if( !this->isReference() )
    {
      if(m_array_1D)
        delete m_array_1D;
      if(m_array_2D)
        delete m_array_2D;
      if(m_array_3D)
        delete m_array_3D;
      if(m_array_4D)
        delete m_array_4D;
    }

    if(m_view_1D)
       delete m_view_1D;
    if(m_view_2D)
      delete m_view_2D;
    if(m_view_3D)
      delete m_view_3D;
    if(m_view_4D)
      delete m_view_4D;

    if(m_data_RW)
      delete [] m_data_RW;

    m_array_1D = 0;
    m_array_2D = 0;
    m_array_3D = 0;
    m_array_4D = 0;
    m_view_1D = 0;
    m_view_2D = 0;
    m_view_3D = 0;
    m_view_4D = 0;

    m_data_RW = 0;
  }


  template <typename T> TensorTemplate<T>::~TensorTemplate()
  {
    cleanup();
  }


  template <typename T> void TensorTemplate<T>::setDataType()
  {
    setDataTypeMain( Tensor::Undefined);
  }

  template <> inline void TensorTemplate<char>::setDataType()
  {
    setDataTypeMain( Tensor::Char);
  }

  template <> inline void TensorTemplate<short>::setDataType()
  {
    setDataTypeMain( Tensor::Short);
  }

  template <> inline void TensorTemplate<int>::setDataType()
  {
    setDataTypeMain( Tensor::Int);
  }

  template <> inline void TensorTemplate<long>::setDataType()
  {
    setDataTypeMain( Tensor::Long);
  }

  template <> inline void TensorTemplate<float>::setDataType()
  {
    setDataTypeMain( Tensor::Float);
  }

  template <> inline void TensorTemplate<double>::setDataType()
  {
    setDataTypeMain( Tensor::Double);
  }


  template <typename T> void
  TensorTemplate<T>::indicesCorrectOrder( long dim0, long dim1, long *result) const
  {
      result[m_dimension_index[0]] = dim0;
      result[m_dimension_index[1]] = dim1;
  }

  template <typename T> void 
  TensorTemplate<T>::indicesCorrectOrder( long dim0, long dim1, long dim2, long *result) const
  {
    result[m_dimension_index[0]] = dim0;
    result[m_dimension_index[1]] = dim1;
    result[m_dimension_index[2]] = dim2;
  }

  template <typename T> void
  TensorTemplate<T>::indicesCorrectOrder( long dim0, long dim1, long dim2, long dim3, long *result) const
  {
    result[m_dimension_index[0]] = dim0;
    result[m_dimension_index[1]] = dim1;
    result[m_dimension_index[2]] = dim2;
    result[m_dimension_index[3]] = dim3;
  }


  template <typename T> void
  TensorTemplate<T>::initDimensionIndex()
  {
    for( size_t i = 0; i < 4 ; i++)
      m_dimension_index[i] = i;
  }


  template <typename T> bool
  TensorTemplate<T>::setDimensionIndex(const size_t dimensions[4])
  {
    for(size_t i=0; i<4; i++)
      m_dimension_index[i] = dimensions[i];
    return true;
  }

  template <typename T> bool
  TensorTemplate<T>::removeDimensionIndex(const size_t dimensionRemoved)
  {
    size_t j = m_dimension_index[dimensionRemoved];
    // Remove the dimension by shifting values after this index
    for(size_t i=dimensionRemoved; i<3; i++)
      m_dimension_index[i] = m_dimension_index[i+1];

    // Decrease values which were higher than m_dimension_index[dimensionRemoved]
    for(size_t i=0; i<3; i++)
    {
      if( m_dimension_index[i] > j)
        m_dimension_index[i]--;
    }

    return true;
  }

/*
  template <typename T> void
  Tensor<T>::initViewRelatedArrays()
  {
    for( size_t i = 0; i < 4 ; i++)
    {
      m_view_first_index[i] = 0;
      m_view_size[i] = 0;
    }
  }
*/

  template <typename T> int
  TensorTemplate<T>::nDimension() const
  {
    return m_n_dimensions;
  }


  template <typename T> long
  TensorTemplate<T>::size(int dimension) const
  {
    switch(m_n_dimensions)
    {
      case 1: return m_view_1D->extent(m_dimension_index[dimension]);
      case 2: return m_view_2D->extent(m_dimension_index[dimension]);
      case 3: return m_view_3D->extent(m_dimension_index[dimension]);
      case 4: return m_view_4D->extent(m_dimension_index[dimension]);
      default: return 0;
    }
  }

  
  template <typename T> long
  TensorTemplate<T>::sizeAll() const
  {
    if( m_n_dimensions == 0)
      return 0;

    long res = 1;
    for( size_t i=0; i<m_n_dimensions; i++)
      res *= size(i);

    return res;
  }

 
  template <typename T> void 
  TensorTemplate<T>::setTensor( const TensorTemplate<T> *src)
  {
    // Delete previous content of this Tensor if required
    cleanup();

    // Add a reference to the src Tensor content
    m_is_reference = true;
    m_n_dimensions = src->nDimension();
    m_n_dimensions_array = src->getDimensionArray();
    setDimensionIndex(src->getDimensionIndex());
    switch(m_n_dimensions_array) 
    {
      case 1:
        m_array_1D = src->getArray1D();
        break;
      case 2: 
        m_array_2D = src->getArray2D();
        break;
      case 3: 
        m_array_3D = src->getArray3D();
        break;
      case 4: 
        m_array_4D = src->getArray4D();
        break;
    } 
    switch(m_n_dimensions) 
    {
      case 1:
        m_view_1D = new Tensor_1D_Type((*src->getView1D())( blitz::Range::all()));
        break;
      case 2: 
        m_view_2D = new Tensor_2D_Type((*src->getView2D())( blitz::Range::all(), blitz::Range::all()));
        break;
      case 3: 
        m_view_3D = new Tensor_3D_Type((*src->getView3D())( blitz::Range::all(), blitz::Range::all(), blitz::Range::all()));
        break;
      case 4: 
        m_view_4D = new Tensor_4D_Type((*src->getView4D())( blitz::Range::all(), blitz::Range::all(), blitz::Range::all(), blitz::Range::all()));
        break;
    }
  }


//  template <typename T> 
  template <typename T> template <typename U> void 
  TensorTemplate<T>::copy( const TensorTemplate<U> *src)
  {
    // Delete previous content of this Tensor if required
    // Not done like that in previous Torch5spro implementation!
    // cleanup();

    // If Tensor is not a reference, this allows to resize it
    if( !m_is_reference )
      resizeAs( *src);


    if( this->nDimension() != src->nDimension() )
      std::cerr << "Tensor::copy(): cannot copy tensors with different number of dimensions." << std::endl;
   
    for( long d = 0; d < this->nDimension(); d++)
    {
      if(size(d) != src->size(d))
      {
        std::cerr << "Tensor::copy(): cannot copy tensors of different sizes ";
        std::cerr << "(Dimension of index " << d << ": ";
        std::cerr << size(d) << " != " << src->size(d) << ")." << std::endl;
      }
    }
        
    switch(m_n_dimensions)
    {
      case 1: 
        for( int i = 0; i < size(0); i++)
          set( i, (T)(src->get(i)) );
        break;
      case 2: 
        for( int i = 0; i < size(0); i++)
          for( int j = 0; j < size(1); j++)
            set( i, j, (T)(src->get( i, j)) );
        break;
      case 3:
        for( int i = 0; i < size(0); i++)
          for( int j = 0; j < size(1); j++)
            for( int k = 0; k < size(2); k++)
              set( i, j, k, (T)(src->get( i, j, k)) );
        break;
      case 4:
        for( int i = 0; i < size(0); i++)
          for( int j = 0; j < size(1); j++)
            for( int k = 0; k < size(2); k++)
              for( int l = 0; l < size(3); l++)
                set( i, j, k, l, (T)(src->get( i, j, k, l)) );
        break;
      default:
        break;
    }


/*
    m_is_reference = false;
    m_n_dimensions_array = src->nDimension();
    m_n_dimensions = m_n_dimensions_array;
    initDimensionIndex();
//    setDimensionIndex(src->getDimensionIndex());
    switch(m_n_dimensions)
    {
      case 1: 
        m_array_1D = new Tensor_1D_Type();
        m_array_1D->resize(boost::extents[src->size(0)]);
        m_view_1D = new View_1D_Type((*m_array_1D)[ boost::indices[range()] ]);
        break;
      case 2: 
        m_array_2D = new Tensor_2D_Type();
        m_array_2D->resize(boost::extents[src->size(0)][src->size(1)]);
        m_view_2D = new View_2D_Type((*m_array_2D)[ boost::indices[range()][range()] ]);
        break;
      case 3:
        m_array_3D = new Tensor_3D_Type();
        m_array_3D->resize(boost::extents[src->size(0)][src->size(1)][src->size(2)]);
        m_view_3D = new View_3D_Type((*m_array_3D)[ boost::indices[range()][range()][range()] ]);
        break;
      case 4:
        m_array_4D = new Tensor_4D_Type();
        m_array_4D->resize(boost::extents[src->size(0)][src->size(1)][src->size(2)][src->size(3)]);
        m_view_4D = new View_4D_Type((*m_array_4D)[ boost::indices[range()][range()][range()][range()] ]);
        break;
    }


//    if( !m_is_reference )
//    TODO: Take care of m_dimension_index!!!
    {
//      resizeAs( *src);
      switch(m_n_dimensions)
      {
      case 1:
        for(long i = 0; i<size(0); i++)
          (*m_array_1D)[i] = (T)((*src->getView1D())[i]); 
//        std::copy(src->m_array_1D->data(),src->m_array_1D->data()+src->m_array_1D->num_elements(), m_array_1D->data());
        break;
      case 2:
        for(long i = 0; i<size(0); i++)
          for(long j = 0; j<size(1); j++)
          {
            long* rdim = src->indicesCorrectOrder(i, j);
            (*m_array_2D)[i][j] = (T)((*src->getView2D())[rdim[0]][rdim[1]]);
            delete [] rdim;
          }
//        std::copy(src->m_array_2D->data(),src->m_array_2D->data()+src->m_array_2D->num_elements(), m_array_2D->data());
        break;
      case 3:
        for(long i = 0; i<size(0); i++)
          for(long j = 0; j<size(1); j++)
            for(long k = 0; k<size(2); k++)
            {
              long* rdim = src->indicesCorrectOrder(i, j, k);
              (*m_array_3D)[i][j][k] = (T)((*src->getView3D())[rdim[0]][rdim[1]][rdim[2]]);
              delete [] rdim;
            }
//        std::copy(src->m_array_3D->data(),src->m_array_3D->data()+src->m_array_3D->num_elements(), m_array_3D->data());
        break;
      case 4:
        for(long i = 0; i<size(0); i++)
          for(long j = 0; j<size(1); j++)
            for(long k = 0; k<size(2); k++)
              for(long l = 0; l<size(3); l++)
              {
                long* rdim = src->indicesCorrectOrder(i, j, k, l);
                (*m_array_4D)[i][j][k][l] = (T)((*src->getView4D())[rdim[0]][rdim[1]][rdim[2]][rdim[3]]);
                delete [] rdim;
              }
//        std::copy(src->m_array_4D->data(),src->m_array_4D->data()+src->m_array_4D->num_elements(), m_array_4D->data());
        break;
      default:
        break;
      }
    }
*/
  }


  template <typename T> void 
  TensorTemplate<T>::transpose( const TensorTemplate<T> *src, int dimension1, int dimension2)
  {

    this->setTensor( src );
    if( dimension1 >= this->nDimension() || dimension2 >= this->nDimension() )
      std::cerr << "Error: Cannot transpose non existing dimensions" << std::endl;
      

    if( dimension1 != dimension2)
    { 
      size_t tmp_index = m_dimension_index[dimension1];
      m_dimension_index[dimension1] = m_dimension_index[dimension2];
      m_dimension_index[dimension2] = tmp_index;
    }
  }


  template <typename T> void 
  TensorTemplate<T>::print( const char *name) const
  {
    if(name != NULL) 
      std::cout << "Tensor " << name << ":" << std::endl;;
    Tprint();
  }


  template <typename T> void 
  TensorTemplate<T>::sprint( const char *name, ...) const
  {
    if(name != NULL)
    {
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

  template <typename T> void
  TensorTemplate<T>::Tprint() const
  {
    if(nDimension() == 0) std::cout << "Oups !! I can't print a 0D tensor :-(" << std::endl;
    else if(nDimension() == 1)
    {
      for(int y = 0 ; y < size(0) ; y++)
      {
        T v = get( y);
        std::cout << v << " ";
      }
      std::cout << std::endl;
    }
    else if(nDimension() == 2)
    {
      for(int y = 0 ; y < size(0) ; y++)
      {
        for(int x = 0 ; x < size(1) ; x++)
        {
          T v = get( y, x);
          std::cout << v << " ";
        }
        std::cout << std::endl;
      }
    }
    else if(nDimension() == 3)
    {
      for(int y = 0 ; y < size(0) ; y++)
      {
        for(int x = 0 ; x < size(1) ; x++)
        {
          std::cout << " (";
          for(int z = 0 ; z < size(2) ; z++)
          {
            T v = get( y, x, z);
            std::cout << v << " ";
          }
          std::cout << ") ";
        }
        std::cout << std::endl;
      }
    }
    else if(nDimension() == 4) std::cout << "Sorry I don't know how to print a 4D tensor :-(" << std::endl;
  }


  template <> inline void
  TensorTemplate<char>::Tprint() const
  {
    if(nDimension() == 0) std::cout << "Oups !! I can't print a 0D tensor :-(" << std::endl;
    else if(nDimension() == 1)
    {
      for(int y = 0 ; y < size(0) ; y++)
      {
        char v = get( y);
        std::cout << static_cast<char>(v) << " ";
      }
      std::cout << std::endl;
    }
    else if(nDimension() == 2)
    {
      for(int y = 0 ; y < size(0) ; y++)
      {
        for(int x = 0 ; x < size(1) ; x++)
        {
          char v = get( y, x);
          std::cout << static_cast<char>(v) << " ";
        }
        std::cout << std::endl;
      }
    }
    else if(nDimension() == 3)
    {
      for(int y = 0 ; y < size(0) ; y++)
      {
        for(int x = 0 ; x < size(1) ; x++)
        {
          std::cout << " (";
          for(int z = 0 ; z < size(2) ; z++)
          {
            char v = get( y, x, z);
            std::cout << static_cast<char>(v) << " ";
          }
          std::cout << " )";
        }
        std::cout << std::endl;
      }
    }
    else if(nDimension() == 4) std::cout << "Sorry I don't know how to print a 4D tensor :-(" << std::endl;
  }


  template <typename T> template <typename U> void
  TensorTemplate<T>::resizeAs( const TensorTemplate<U> &src)
  {
    // Delete previous content of this Tensor if required
    cleanup();

    // Copy the view in a new array!!!
    m_is_reference = false;
    m_n_dimensions_array = src.nDimension();
    m_n_dimensions = m_n_dimensions_array;
    initDimensionIndex();
//    setDimensionIndex(src.getDimensionIndex());
    switch(m_n_dimensions)
    {
      case 1: 
        m_array_1D = new Tensor_1D_Type((int)src.size(0));
        m_view_1D = new Tensor_1D_Type((*m_array_1D)( blitz::Range::all()));
        break;
      case 2: 
        m_array_2D = new Tensor_2D_Type((int)src.size(0), (int)src.size(1));
        m_view_2D = new Tensor_2D_Type((*m_array_2D)( blitz::Range::all(), blitz::Range::all()));
        break;
      case 3:
        m_array_3D = new Tensor_3D_Type((int)src.size(0), (int)src.size(1), (int)src.size(2));
        m_view_3D = new Tensor_3D_Type((*m_array_3D)( blitz::Range::all(), blitz::Range::all(), blitz::Range::all()));
        break;
      case 4:
        m_array_4D = new Tensor_4D_Type((int)src.size(0), (int)src.size(1), (int)src.size(2), (int)src.size(3));
        m_view_4D = new Tensor_4D_Type((*m_array_4D)( blitz::Range::all(), blitz::Range::all(), blitz::Range::all(), blitz::Range::all()));
        break;
    }
  }

  template <typename T> void
  TensorTemplate<T>::resize( long size0)
  {
    if( m_is_reference )
    {
      std::cerr << "TensorTemplate::resize(): cannot resize a reference Tensor" << std::endl;
      exit(-1);
    }
    else
    {
      // Delete previous content of this Tensor if required
      cleanup();

      // Initialize it
      m_n_dimensions_array = 1;
      m_n_dimensions = m_n_dimensions_array;
      initDimensionIndex();

      m_array_1D = new Tensor_1D_Type((int)size0);
      m_view_1D = new Tensor_1D_Type((*m_array_1D)( blitz::Range::all()));
    }
  }

  template <typename T> void
  TensorTemplate<T>::resize( long size0, long size1)
  {
    if( m_is_reference )
    {
      std::cerr << "TensorTemplate::resize(): cannot resize a reference Tensor" << std::endl;
      exit(-1);
    }
    else
    {
      // Delete previous content of this Tensor if required
      cleanup();

      // Initialize it
      m_n_dimensions_array = 2;
      m_n_dimensions = m_n_dimensions_array;
      initDimensionIndex();

      m_array_2D = new Tensor_2D_Type((int)size0, (int)size1);
      m_view_2D = new Tensor_2D_Type((*m_array_2D)( blitz::Range::all(), blitz::Range::all()));
    }
  }

  template <typename T> void
  TensorTemplate<T>::resize( long size0, long size1, long size2)
  {
    if( m_is_reference )
    {
      std::cerr << "TensorTemplate::resize(): cannot resize a reference Tensor" << std::endl;
      exit(-1);
    }
    else
    {
      // Delete previous content of this Tensor if required
      cleanup();

      // Initialize it
      m_n_dimensions_array = 3;
      m_n_dimensions = m_n_dimensions_array;
      initDimensionIndex();

      m_array_3D = new Tensor_3D_Type((int)size0, (int)size1, (int)size2);
      m_view_3D = new Tensor_3D_Type((*m_array_3D)( blitz::Range::all(), blitz::Range::all(), blitz::Range::all()));
    }
  }

  template <typename T> void
  TensorTemplate<T>::resize( long size0, long size1, long size2, long size3)
  {
    if( m_is_reference )
    {
      std::cerr << "TensorTemplate::resize(): cannot resize a reference Tensor" << std::endl;
      exit(-1);
    }
    else
    {
      // Delete previous content of this Tensor if required
      cleanup();

      // Initialize it
      m_is_reference = false;
      m_n_dimensions_array = 4;
      m_n_dimensions = m_n_dimensions_array;
      initDimensionIndex();

      m_array_4D = new Tensor_4D_Type((int)size0, (int)size1, (int)size2, (int)size3);
      m_view_4D = new Tensor_4D_Type((*m_array_4D)( blitz::Range::all(), blitz::Range::all(), blitz::Range::all(), blitz::Range::all()));
    }
  }


  template <typename T> void
  TensorTemplate<T>::set(long dim0, char value)
  {
    (*m_view_1D)((int)dim0) = (T)value; 
  }

  template <typename T> void
  TensorTemplate<T>::set(long dim0, long dim1, char value)
  {
    long rdim[2];
    indicesCorrectOrder(dim0, dim1, rdim);
    (*m_view_2D)((int)rdim[0],(int)rdim[1]) = (T)value;
  }

  template <typename T> void
  TensorTemplate<T>::set(long dim0, long dim1, long dim2, char value)
  {
    long rdim[3];
    indicesCorrectOrder(dim0, dim1, dim2, rdim);
    (*m_view_3D)((int)rdim[0],(int)rdim[1],(int)rdim[2]) = (T)value;
  }

  template <typename T> void
  TensorTemplate<T>::set(long dim0, long dim1, long dim2, long dim3, char value)
  {
    long rdim[4];
    indicesCorrectOrder(dim0, dim1, dim2, dim3, rdim);
    (*m_view_4D)((int)rdim[0],(int)rdim[1],(int)rdim[2],(int)rdim[3]) = (T)value;
  }

  template <typename T> void
  TensorTemplate<T>::set(long dim0, short value)
  {
    (*m_view_1D)((int)dim0) = (T)value; 
  }

  template <typename T> void
  TensorTemplate<T>::set(long dim0, long dim1, short value)
  {
    long rdim[2];
    indicesCorrectOrder(dim0, dim1, rdim);
    (*m_view_2D)((int)rdim[0],(int)rdim[1]) = (T)value;
  }

  template <typename T> void
  TensorTemplate<T>::set(long dim0, long dim1, long dim2, short value)
  {
    long rdim[3];
    indicesCorrectOrder(dim0, dim1, dim2, rdim);
    (*m_view_3D)((int)rdim[0],(int)rdim[1],(int)rdim[2]) = (T)value;
  }

  template <typename T> void
  TensorTemplate<T>::set(long dim0, long dim1, long dim2, long dim3, short value)
  {
    long rdim[4];
    indicesCorrectOrder(dim0, dim1, dim2, dim3, rdim);
    (*m_view_4D)((int)rdim[0],(int)rdim[1],(int)rdim[2],(int)rdim[3]) = (T)value;
  }

  template <typename T> void
  TensorTemplate<T>::set(long dim0, int value)
  {
    (*m_view_1D)((int)dim0) = (T)value; 
  }

  template <typename T> void
  TensorTemplate<T>::set(long dim0, long dim1, int value)
  {
    long rdim[2];
    indicesCorrectOrder(dim0, dim1, rdim);
    (*m_view_2D)((int)rdim[0],(int)rdim[1]) = (T)value;
  }

  template <typename T> void
  TensorTemplate<T>::set(long dim0, long dim1, long dim2, int value)
  {
    long rdim[3];
    indicesCorrectOrder(dim0, dim1, dim2, rdim);
    (*m_view_3D)((int)rdim[0],(int)rdim[1],(int)rdim[2]) = (T)value;
  }

  template <typename T> void
  TensorTemplate<T>::set(long dim0, long dim1, long dim2, long dim3, int value)
  {
    long rdim[4];
    indicesCorrectOrder(dim0, dim1, dim2, dim3, rdim);
    (*m_view_4D)((int)rdim[0],(int)rdim[1],(int)rdim[2],(int)rdim[3]) = (T)value;
  }

  template <typename T> void
  TensorTemplate<T>::set(long dim0, long value)
  {
    (*m_view_1D)((int)dim0) = (T)value; 
  }

  template <typename T> void
  TensorTemplate<T>::set(long dim0, long dim1, long value)
  {
    long rdim[2];
    indicesCorrectOrder(dim0, dim1, rdim);
    (*m_view_2D)((int)rdim[0],(int)rdim[1]) = (T)value;
  }

  template <typename T> void
  TensorTemplate<T>::set(long dim0, long dim1, long dim2, long value)
  {
    long rdim[3];
    indicesCorrectOrder(dim0, dim1, dim2, rdim);
    (*m_view_3D)((int)rdim[0],(int)rdim[1],(int)rdim[2]) = (T)value;
  }

  template <typename T> void
  TensorTemplate<T>::set(long dim0, long dim1, long dim2, long dim3, long value)
  {
    long rdim[4];
    indicesCorrectOrder(dim0, dim1, dim2, dim3, rdim);
    (*m_view_4D)((int)rdim[0],(int)rdim[1],(int)rdim[2],(int)rdim[3]) = (T)value;
  }

  template <typename T> void
  TensorTemplate<T>::set(long dim0, float value)
  {
    (*m_view_1D)((int)dim0) = (T)value; 
  }

  template <typename T> void
  TensorTemplate<T>::set(long dim0, long dim1, float value)
  {
    long rdim[2];
    indicesCorrectOrder(dim0, dim1, rdim);
    (*m_view_2D)((int)rdim[0],(int)rdim[1]) = (T)value;
  }

  template <typename T> void
  TensorTemplate<T>::set(long dim0, long dim1, long dim2, float value)
  {
    long rdim[3];
    indicesCorrectOrder(dim0, dim1, dim2, rdim);
    (*m_view_3D)((int)rdim[0],(int)rdim[1],(int)rdim[2]) = (T)value;
  }

  template <typename T> void
  TensorTemplate<T>::set(long dim0, long dim1, long dim2, long dim3, float value)
  {
    long rdim[4];
    indicesCorrectOrder(dim0, dim1, dim2, dim3, rdim);
    (*m_view_4D)((int)rdim[0],(int)rdim[1],(int)rdim[2],(int)rdim[3]) = (T)value;
  }

  template <typename T> void
  TensorTemplate<T>::set(long dim0, double value)
  {
    (*m_view_1D)((int)dim0) = (T)value; 
  }

  template <typename T> void
  TensorTemplate<T>::set(long dim0, long dim1, double value)
  {
    long rdim[2];
    indicesCorrectOrder(dim0, dim1, rdim);
    (*m_view_2D)((int)rdim[0],(int)rdim[1]) = (T)value;
  }

  template <typename T> void
  TensorTemplate<T>::set(long dim0, long dim1, long dim2, double value)
  {
    long rdim[3];
    indicesCorrectOrder(dim0, dim1, dim2, rdim);
    (*m_view_3D)((int)rdim[0],(int)rdim[1],(int)rdim[2]) = (T)value;
  }

  template <typename T> void
  TensorTemplate<T>::set(long dim0, long dim1, long dim2, long dim3, double value)
  {
    long rdim[4];
    indicesCorrectOrder(dim0, dim1, dim2, dim3, rdim);
    (*m_view_4D)((int)rdim[0],(int)rdim[1],(int)rdim[2],(int)rdim[3]) = (T)value;
  }



  template <typename T> T
  TensorTemplate<T>::sum() const
  {
    T res = 0;
    switch(m_n_dimensions)
    {
      case 1:
        for(int i = 0; i<size(0); i++)
          res += (*m_view_1D)(i);
//        std::accumulate(m_array_1D->data(), m_array_1D->data()+m_array_1D->num_elements(), 0);
        break;
      case 2:
        for(int i = 0; i<size(0); i++)
          for(int j = 0; j<size(1); j++)
            res += (*m_view_2D)(i,j);
//        std::accumulate(m_array_2D->data(), m_array_2D->data()+m_array_2D->num_elements(), 0);
        break;
      case 3:
        for(int i = 0; i<size(0); i++)
          for(int j = 0; j<size(1); j++)
            for(int k = 0; k<size(2); k++)
              res += (*m_view_3D)(i,j,k);
//        std::accumulate(m_array_3D->data(), m_array_3D->data()+m_array_3D->num_elements(), 0);
        break;
      case 4:
        for(int i = 0; i<size(0); i++)
          for(int j = 0; j<size(1); j++)
            for(int k = 0; k<size(2); k++)
              for(int l = 0; l<size(3); l++)
                res += (*m_view_4D)(i,j,k,l);
//        std::accumulate(m_array_4D->data(), m_array_4D->data()+m_array_4D->num_elements(), 0);
      default:
        break;
    }
    return res;
  }


  template <typename T> T
  TensorTemplate<T>::get(long dim0) const
  {
    long ndim0;
    long ndim1;
    long ndim2;
    long ndim3;
    switch(m_n_dimensions)
    {
      case 1:
        return (*m_view_1D)((int)dim0);
        break;
      case 2:
        ndim0=dim0 % size(0);
        ndim1=dim0 / size(0);
        return get(ndim0, ndim1);
        break;
      case 3:
        ndim2=dim0 / (size(0)*size(1));
        dim0 -= (size(0)*size(1))*ndim2;
        ndim1=dim0 / size(0);
        ndim0=dim0 % size(0);
        return get(ndim0, ndim1, ndim2);
        break;
      case 4:
        ndim3=dim0 / (size(0)*size(1)*size(2));
        dim0 -= (size(0)*size(1)*size(2))*ndim3;
        ndim2=dim0 / (size(0)*size(1));
        dim0 -= (size(0)*size(1))*ndim2;
        ndim1=dim0 / size(0);
        ndim0=dim0 % size(0);
        return get(ndim0, ndim1, ndim2, ndim3);
        break;
      default:
        std::cerr << "Tensor::get(): do not know what to do with " << m_n_dimensions << " dimensions." << std::endl;
        return 0;
        break;
    }
  }

  template <typename T> T
  TensorTemplate<T>::get(long dim0, long dim1) const
  {
    long rdim[2];
    indicesCorrectOrder(dim0, dim1, rdim);
    T res = (*m_view_2D)((int)rdim[0], (int)rdim[1]);
    return res;
  }

  template <typename T> T
  TensorTemplate<T>::get(long dim0, long dim1, long dim2) const
  {
    long rdim[3];
    indicesCorrectOrder(dim0, dim1, dim2, rdim);
    T res = (*m_view_3D)((int)rdim[0], (int)rdim[1], (int)rdim[2]);
    return res;
  }

  template <typename T> T
  TensorTemplate<T>::get(long dim0, long dim1, long dim2, long dim3) const
  {
    long rdim[4];
    indicesCorrectOrder(dim0, dim1, dim2, dim3, rdim);
    T res = (*m_view_4D)((int)rdim[0], (int)rdim[1], (int)rdim[2], (int)rdim[3]);
    return res;
  }


  template <typename T> T&
  TensorTemplate<T>::operator()(long dim0)
  {
    long ndim0;
    long ndim1;
    long ndim2;
    long ndim3;
    switch(m_n_dimensions)
    {
      case 1:
        return (*m_view_1D)((int)dim0);
        break;
      case 2:
        ndim0=dim0 % size(0);
        ndim1=dim0 / size(0);
        return (*this)(ndim0, ndim1);
        break;
      case 3:
        ndim2=dim0 / (size(0)*size(1));
        dim0 -= (size(0)*size(1))*ndim2;
        ndim1=dim0 / size(0);
        ndim0=dim0 % size(0);
        return (*this)(ndim0, ndim1, ndim2);
        break;
      case 4:
        ndim3=dim0 / (size(0)*size(1)*size(2));
        dim0 -= (size(0)*size(1)*size(2))*ndim3;
        ndim2=dim0 / (size(0)*size(1));
        dim0 -= (size(0)*size(1))*ndim2;
        ndim1=dim0 / size(0);
        ndim0=dim0 % size(0);
        return (*this)(ndim0, ndim1, ndim2, ndim3);
        break;
      default:
        std::cerr << "Error Tensor::(): do not know what to do with " << m_n_dimensions << " dimensions." << std::endl;
        exit(-1);
    }
  }

  template <typename T> T&
  TensorTemplate<T>::operator()(long dim0, long dim1)
  {
    long rdim[2];
    indicesCorrectOrder(dim0, dim1, rdim);
    return (*m_view_2D)((int)rdim[0],(int)rdim[1]);
  }

  template <typename T> T&
  TensorTemplate<T>::operator()(long dim0, long dim1, long dim2)
  {
    long rdim[3];
    indicesCorrectOrder(dim0, dim1, dim2, rdim);
    return (*m_view_3D)((int)rdim[0],(int)rdim[1],(int)rdim[2]);
  }

  template <typename T> T&
  TensorTemplate<T>::operator()(long dim0, long dim1, long dim2, long dim3)
  {
    long rdim[4];
    indicesCorrectOrder(dim0, dim1, dim2, dim3, rdim);
    return (*m_view_4D)((int)rdim[0],(int)rdim[1],(int)rdim[2],(int)rdim[3]);
  }


  template <typename T> const T&
  TensorTemplate<T>::operator()(long dim0) const
  {
    long ndim0;
    long ndim1;
    long ndim2;
    long ndim3;
    switch(m_n_dimensions)
    {
      case 1:
        return (*m_view_1D)((int)dim0);
        break;
      case 2:
        ndim0=dim0 % size(0);
        ndim1=dim0 / size(0);
        return (*this)(ndim0, ndim1);
        break;
      case 3:
        ndim2=dim0 / (size(0)*size(1));
        dim0 -= (size(0)*size(1))*ndim2;
        ndim1=dim0 / size(0);
        ndim0=dim0 % size(0);
        return (*this)(ndim0, ndim1, ndim2);
        break;
      case 4:
        ndim3=dim0 / (size(0)*size(1)*size(2));
        dim0 -= (size(0)*size(1)*size(2))*ndim3;
        ndim2=dim0 / (size(0)*size(1));
        dim0 -= (size(0)*size(1))*ndim2;
        ndim1=dim0 / size(0);
        ndim0=dim0 % size(0);
        return (*this)(ndim0, ndim1, ndim2, ndim3);
        break;
      default:
        std::cerr << "Tensor::(): do not know what to do with " << m_n_dimensions << " dimensions." << std::endl;
        exit(-1);
        break;
    }
  }

  template <typename T> const T&
  TensorTemplate<T>::operator()(long dim0, long dim1) const
  {
    long rdim[2];
    indicesCorrectOrder(dim0, dim1, rdim);
    return (*m_view_2D)((int)rdim[0],(int)rdim[1]); 
  }

  template <typename T> const T&
  TensorTemplate<T>::operator()(long dim0, long dim1, long dim2) const
  {
    long rdim[3];
    indicesCorrectOrder(dim0, dim1, dim2, rdim);
    return (*m_view_3D)((int)rdim[0],(int)rdim[1],(int)rdim[2]);
  }

  template <typename T> const T&
  TensorTemplate<T>::operator()(long dim0, long dim1, long dim2, long dim3) const
  {
    long rdim[4];
    indicesCorrectOrder(dim0, dim1, dim2, dim3, rdim);
    return (*m_view_4D)((int)rdim[0],(int)rdim[1],(int)rdim[2],(int)rdim[3]);
  }



  template <typename T> void 
  TensorTemplate<T>::computeDataRW()
  {
    switch(m_n_dimensions)
    {
      case 1:
        for(int i = 0; i<size(0); i++)
          m_data_RW[i] = (*m_view_1D)(i);
        break;
      case 2:
        for(int j = 0; j<size(1); j++)
          for(int i = 0; i<size(0); i++)
            m_data_RW[i+j*m_stride[1]] = (*m_view_2D)(i, j);
        break;
      case 3:
        for(int k = 0; k<size(2); k++)
          for(int j = 0; j<size(1); j++)
            for(int i = 0; i<size(0); i++)
              m_data_RW[i+j*m_stride[1]+k*m_stride[2]] = (*m_view_3D)(i, j, k);
        break;
      case 4:
        for(int l = 0; l<size(3); l++)
          for(int k = 0; k<size(2); k++)
            for(int j = 0; j<size(1); j++)
              for(int i = 0; i<size(0); i++)
                m_data_RW[i+j*m_stride[1]+k*m_stride[2]+l*m_stride[3]] = (*m_view_4D)(i, j, k, l);
      default:
        break;
    }
  }

  template <typename T> const T* 
  TensorTemplate<T>::dataR() const
  {
    TensorTemplate<T>* self = const_cast<TensorTemplate<T>*>(this);

    size_t array_length = sizeAll();
    if( !self->m_data_RW)
      self->m_data_RW = new T[array_length];

    self->computeStride();
    self->computeDataRW();
    return self->m_data_RW;
  }

  template <typename T> T* 
  TensorTemplate<T>::dataW()
  {
    size_t array_length = sizeAll();
    if( !m_data_RW)
      m_data_RW = new T[array_length];

    computeStride();
    computeDataRW();
    return m_data_RW;
  }

 
 
  template <typename T> void 
  TensorTemplate<T>::computeStride()
  {
    switch(m_n_dimensions)
    {
      case 1:
        m_stride[0] = 1;
        m_stride[1] = m_stride[2] = m_stride[3] = 0;
        break;
      case 2:
        m_stride[0] = 1;
        m_stride[1] = size(0);
        m_stride[2] = m_stride[3] = 0;
        break;
      case 3:
        m_stride[0] = 1;
        m_stride[1] = size(0);
        m_stride[2] = size(0)*size(1);
        m_stride[3] = 0;
        break;
      case 4:
        m_stride[0] = 1;
        m_stride[1] = size(0);
        m_stride[2] = size(0)*size(1);
        m_stride[3] = size(0)*size(1)*size(2);
      default:
        break;
    }
  }

  template <typename T> long
  TensorTemplate<T>::stride(int dim) const
  {
    TensorTemplate<T>* self = const_cast<TensorTemplate<T>*>(this);
    self->computeStride();
    return self->m_stride[dim];
  }

  template <typename T> void
  TensorTemplate<T>::resetFromData()
  {
    switch(m_n_dimensions)
    {
      case 1:
        for(int i = 0; i<size(0); i++)
          (*m_view_1D)(i) = m_data_RW[i];
        break;
      case 2:
        for(int j = 0; j<size(1); j++)
          for(int i = 0; i<size(0); i++)
            (*m_view_2D)(i, j) = m_data_RW[i+j*m_stride[1]];
        break;
      case 3:
        for(int k = 0; k<size(2); k++)
          for(int j = 0; j<size(1); j++)
            for(int i = 0; i<size(0); i++)
              (*m_view_3D)(i, j, k) = m_data_RW[i+j*m_stride[1]+k*m_stride[2]];
        break;
      case 4:
        for(int l = 0; l<size(3); l++)
          for(int k = 0; k<size(2); k++)
            for(int j = 0; j<size(1); j++)
              for(int i = 0; i<size(0); i++)
                (*m_view_4D)(i, j, k, l) = m_data_RW[i+j*m_stride[1]+k*m_stride[2]+l*m_stride[3]];
      default:
        break;
    }
  }


  template <typename T> void
  TensorTemplate<T>::narrow( const TensorTemplate<T> *src, int dimension, long firstIndex, long size)
  {
    // Delete previous content of this Tensor if required
    cleanup();

    // Add a reference to the src Tensor content
    m_is_reference = true;
    m_n_dimensions_array = src->getDimensionArray();
    setDimensionIndex(src->getDimensionIndex());
    switch(m_n_dimensions_array) 
    {
      case 1:
        m_array_1D = src->getArray1D();
        break;
      case 2: 
        m_array_2D = src->getArray2D();
        break;
      case 3: 
        m_array_3D = src->getArray3D();
        break;
      case 4: 
        m_array_4D = src->getArray4D();
        break;
    }

    m_n_dimensions = src->nDimension();
    switch(m_n_dimensions) 
    {
      case 1:
        m_view_1D = new Tensor_1D_Type((*src->getView1D())( blitz::Range(firstIndex,firstIndex+size-1)));
        break;
      case 2:
        switch(m_dimension_index[dimension])
        {
          case 0:
            m_view_2D = new Tensor_2D_Type((*src->getView2D())( blitz::Range(firstIndex,firstIndex+size-1), blitz::Range::all()));
            break;
          case 1:
            m_view_2D = new Tensor_2D_Type((*src->getView2D())( blitz::Range::all(), blitz::Range(firstIndex,firstIndex+size-1)));
            break;
        }
        break;
      case 3: 
        switch(m_dimension_index[dimension])
        {
          case 0:
            m_view_3D = new Tensor_3D_Type((*src->getView3D())( blitz::Range(firstIndex,firstIndex+size-1), blitz::Range::all(), blitz::Range::all()));
            break;
          case 1:
            m_view_3D = new Tensor_3D_Type((*src->getView3D())( blitz::Range::all(), blitz::Range(firstIndex,firstIndex+size-1), blitz::Range::all()));
            break;
          case 2:
            m_view_3D = new Tensor_3D_Type((*src->getView3D())( blitz::Range::all(), blitz::Range::all(), blitz::Range(firstIndex,firstIndex+size-1)));
            break;
        }
        break;
      case 4: 
        switch(m_dimension_index[dimension])
        {
          case 0:
            m_view_4D = new Tensor_4D_Type((*src->getView4D())( blitz::Range(firstIndex,firstIndex+size-1), blitz::Range::all(), blitz::Range::all(), blitz::Range::all()));
            break;
          case 1:
            m_view_4D = new Tensor_4D_Type((*src->getView4D())( blitz::Range::all(), blitz::Range(firstIndex,firstIndex+size-1), blitz::Range::all(), blitz::Range::all()));
            break;
          case 2:
            m_view_4D = new Tensor_4D_Type((*src->getView4D())( blitz::Range::all(), blitz::Range::all(), blitz::Range(firstIndex,firstIndex+size-1), blitz::Range::all()));
            break;
          case 3:
            m_view_4D = new Tensor_4D_Type((*src->getView4D())( blitz::Range::all(), blitz::Range::all(), blitz::Range::all(), blitz::Range(firstIndex,firstIndex+size-1)));
            break;
        }
        break;
    } 
  }


  template <typename T> void
  TensorTemplate<T>::select( const TensorTemplate<T> *src, int dimension, long sliceIndex)
  {
    // Delete previous content of this Tensor if required
    cleanup();

    // Add a reference to the src Tensor content
    m_is_reference = true;
    m_n_dimensions_array = src->getDimensionArray();
    setDimensionIndex(src->getDimensionIndex());
    switch(m_n_dimensions_array) 
    {
      case 1:
        m_array_1D = src->getArray1D();
        break;
      case 2: 
        m_array_2D = src->getArray2D();
        break;
      case 3: 
        m_array_3D = src->getArray3D();
        break;
      case 4: 
        m_array_4D = src->getArray4D();
        break;
      default:
        break;
    }

    // TODO: Check Dimension index when "array is large and view is small"
    m_n_dimensions = src->nDimension() - 1;
    switch(m_n_dimensions) 
    {
      case 0:
        std::cerr << "Tensor::select(): Cannot select on a 1D tensor" << std::endl;
        break;
      case 1:
        switch(m_dimension_index[dimension])
        {
          case 0:
            m_view_1D = new Tensor_1D_Type((*src->getView2D())( sliceIndex, blitz::Range::all()));
            break;
          case 1:
            m_view_1D = new Tensor_1D_Type((*src->getView2D())( blitz::Range::all(), sliceIndex));
            break;
          default:
            break;
        }
        break;
      case 2:
        switch(m_dimension_index[dimension])
        {
          case 0:
            m_view_2D = new Tensor_2D_Type((*src->getView3D())( sliceIndex, blitz::Range::all(), blitz::Range::all()));
            break;
          case 1:
            m_view_2D = new Tensor_2D_Type((*src->getView3D())( blitz::Range::all(), sliceIndex, blitz::Range::all()));
            break;
          case 2:
            m_view_2D = new Tensor_2D_Type((*src->getView3D())( blitz::Range::all(), blitz::Range::all(), sliceIndex));
            break;
          default:
            break;
        }
        break;
      case 3:
        switch(m_dimension_index[dimension])
        {
          case 0:
            m_view_3D = new Tensor_3D_Type((*src->getView4D())( sliceIndex, blitz::Range::all(), blitz::Range::all(), blitz::Range::all()));
            break;
          case 1:
            m_view_3D = new Tensor_3D_Type((*src->getView4D())( blitz::Range::all(), sliceIndex, blitz::Range::all(), blitz::Range::all()));
            break;
          case 2:
            m_view_3D = new Tensor_3D_Type((*src->getView4D())( blitz::Range::all(), blitz::Range::all(), sliceIndex, blitz::Range::all()));
            break;
          case 3:
            m_view_3D = new Tensor_3D_Type((*src->getView4D())( blitz::Range::all(), blitz::Range::all(), blitz::Range::all(), sliceIndex));
            break;
          default:
            break;
        }
        break;
      default:
        break;
    } 
    removeDimensionIndex( dimension );
  }

  template <typename T> TensorTemplate<T>*
  TensorTemplate<T>::select( int dimension, long sliceIndex) const
  {
    TensorTemplate<T>* result = new TensorTemplate<T>();
    result->select( this, dimension, sliceIndex);
    return result;
  }

////////// NOT SUPPORTED BY THE WRAPPER
//  template <typename T> template <typename U> void
//  Tensor<T>::unfold(TensorTemplate<U> *src, int dimension, long size, long step)
//  {
//  }


  template <typename T> void 
  TensorTemplate<T>::fill(T val)
  {
    // Notice that there is no need to fill in the "correct order"
    switch(m_n_dimensions)
    {
      case 1:
        for(int i= 0; i<size(0); i++)
          (*m_view_1D)(i) = val; 
//        std::fill(m_array_1D->begin()->begin(),m_array_1D->end()->end()/*+m_array_1D->num_elements()*/, val);  
        break;
      case 2:
        for(long i = 0; i<size(0); i++)
          for(long j = 0; j<size(1); j++)
          {
            long rdim[2];
            indicesCorrectOrder(i, j, rdim);
            (*m_view_2D)((int)rdim[0], (int)rdim[1]) = val;
          }
//        std::fill(m_array_2D->begin(),m_array_2D->end()/*+m_array_2D->num_elements()*/, val);  
        break;
      case 3:
        for(long i = 0; i<size(0); i++)
          for(long j = 0; j<size(1); j++)
            for(long k = 0; k<size(2); k++)
            {
              long rdim[3];
              indicesCorrectOrder(i, j, k, rdim);
              (*m_view_3D)((int)rdim[0], (int)rdim[1], (int)rdim[2]) = val;
            }
//        std::fill(m_array_3D->begin(),m_array_3D->end()/*+m_array_3D->num_elements()*/, val);  
        break;
      case 4:
        for(long i = 0; i<size(0); i++)
          for(long j = 0; j<size(1); j++)
            for(long k = 0; k<size(2); k++)
              for(long l = 0; l<size(3); l++)
              {
                long rdim[4];
                indicesCorrectOrder(i, j, k, l, rdim);
                (*m_view_4D)((int)rdim[0], (int)rdim[1], (int)rdim[2], (int)rdim[3]) = val;
              }
//        std::fill(m_array_4D->begin(),m_array_4D->end()/*+m_array_4D->num_elements()*/, val);  
        break;
      default:
        break;
    }
  }

}

#endif
