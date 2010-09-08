/**
  * @file TensorWrapper.h
  * @author <a href="mailto:Laurent.El-Shafey@idiap.ch">Laurent El Shafey</a> 
  *
  * @brief Wrapper from Boost::Multi_Array to former Tensor class
  */

#include <cstdlib> //exit function
#include <string>
#include <iostream>

namespace Torch {

  /**
   * @brief The Generic Tensor class 
   * (Wrapper from boost::multi_array to the former Torch5spro Tensor class)
   * unfold() is not supported
   */
  class Tensor {

    public:
      /**
       * Enumeration of the possible types of a Tensor
       * The Undefined type was added in comparison to the former Torch5spro
       * Tensor class
       */
      enum Type
      {
        Char,
        Short,
        Int,
        Long,
        Float,
        Double,
        Undefined
      };

      /**
       * Return the number of dimensions of a tensor.
       */
      virtual int nDimension() const = 0;

      /**
       * Return the number of elements inside a particular dimension.
       */
      virtual long size(int dimension_) const = 0;

      /**
       * Get the number of elements (over all dimensions)
       */
      virtual long sizeAll() const = 0;

      /**
       * Set the tensor from another tensor (same type) - this will create a
       * reference
       */
      virtual void setTensor( const Tensor *src);

      /**
       * Copy the tensor from another tensor (copy of any type) - this will
       * make a real copy of the tensor values
       */
      virtual void copy(const Tensor *src);

      /**
       * Transpose 2 dimensions of a tensor
       */
      virtual void transpose(const Tensor *src, int dimension1_, 
          int dimension2_);

      /**
       * Narrow a tensor along dimension #dimension_# starting at slice
       * #firstIndex_# and of #size_# slices
       */
      virtual void narrow(const Tensor *src, int dimension_, long firstIndex_,
          long size_);

      // select a tensor along dimension #dimension_# at slice #sliceIndex_#
      virtual void select(const Tensor *src, int dimension_, long sliceIndex_);

      // select a new tensor along dimension #dimension_# at slice
      // #sliceIndex_#
      virtual Tensor* select(int dimension_, long sliceIndex_) const;

      /**
       * Print the tensor
       */
      virtual void print(const char *name = NULL) const = 0;

      /**
       * Print the tensor
       */
      virtual void sprint(const char *name, ...) const = 0;

      /**
       * Test if the tensor is a reference tensor
       */
      virtual bool isReference() const = 0;

      /**
       * Resize a Tensor
       */
      virtual void resize(long dim0_) = 0;
      virtual void resize(long dim0_, long dim1_) = 0;
      virtual void resize(long dim0_, long dim1_, long dim2_) = 0;
      virtual void resize(long dim0_, long dim1_, long dim2_, long dim3_) = 0;


      /**
       * Set a value of a Tensor
       */
      virtual void set(long, char) = 0;
      virtual void set(long, long, char) = 0;
      virtual void set(long, long, long, char) = 0;
      virtual void set(long, long, long, long, char) = 0;

      virtual void set(long, short) = 0;
      virtual void set(long, long, short) = 0;
      virtual void set(long, long, long, short) = 0;
      virtual void set(long, long, long, long, short) = 0;

      virtual void set(long, int) = 0;
      virtual void set(long, long, int) = 0;
      virtual void set(long, long, long, int) = 0;
      virtual void set(long, long, long, long, int) = 0;

      virtual void set(long, long) = 0;
      virtual void set(long, long, long) = 0;
      virtual void set(long, long, long, long) = 0;
      virtual void set(long, long, long, long, long) = 0;

      virtual void set(long, float) = 0;
      virtual void set(long, long, float) = 0;
      virtual void set(long, long, long, float) = 0;
      virtual void set(long, long, long, long, float) = 0;

      virtual void set(long, double) = 0;
      virtual void set(long, long, double) = 0;
      virtual void set(long, long, long, double) = 0;
      virtual void set(long, long, long, long, double) = 0;

      /**
       * Destructor
       */
      virtual ~Tensor() {}

      /**
       * Return the datatype of the Tensor
       */
      Tensor::Type getDatatype() const { return m_datatype; }

      /**
       *  Get the size of an element
       */
      virtual int typeSize() const;

      const void* dataR() const;
      void*       dataW();
      void        resetFromData();
      long        stride(int dim) const;


      void        raiseError(std::string) const; 
      void        raiseFatalError(std::string) const;


    protected:
      /**
       * Set the datatype of the Tensor
       * @param value the datatype which will be set
       */
      inline void setDataTypeMain(Tensor::Type value) { m_datatype = value; }


    private:
      /**
       * Datatype of the Tensor
       */
      Tensor::Type m_datatype;
  };

}

#if TORCH5SPRO_TENSOR_TYPE == 2
#include "core/TensorBoostTemplate.h"
#include "core/TensorBoostTemplate_impl.h"
#elif TORCH5SPRO_TENSOR_TYPE == 3
#include "core/TensorBlitzTemplate.h"
#include "core/TensorBlitzTemplate_impl.h"
#elif TORCH5SPRO_TENSOR_TYPE == 4
#include "core/TensorBlitzTemplate2.h"
#endif

namespace Torch {
  typedef Torch::TensorTemplate<char>    CharTensor;
  typedef Torch::TensorTemplate<short>   ShortTensor;
  typedef Torch::TensorTemplate<int>     IntTensor;
  typedef Torch::TensorTemplate<long>    LongTensor;
  typedef Torch::TensorTemplate<float>   FloatTensor;
  typedef Torch::TensorTemplate<double>  DoubleTensor;
}

/**
 * @}
 */
