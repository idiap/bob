/**
 * @file src/cxx/database/database/TensorFileHeader.h
 * @author <a href="mailto:Laurent.El-Shafey@idiap.ch">Laurent El Shafey</a> 
 *
 * @brief This class defines an header for storing multiarrays into
 * .tensor files.
 */

#ifndef TORCH_DATABASE_TENSORFILEHEADER_H
#define TORCH_DATABASE_TENSORFILEHEADER_H

#include <fstream>
#include <blitz/array.h>
#include "core/array_type.h"

namespace Torch { namespace database { 

  // TensorType
  enum TensorType
  {
    Char,
    Short,
    Int,
    Long,
    Float,
    Double
  };

  TensorType arrayTypeToTensorType(Torch::core::array::ElementType eltype);
  Torch::core::array::ElementType tensorTypeToArrayType(Torch::database::TensorType tensortype);

  namespace detail {
    /**
     * The Header for storing arrays into binary files. Please note that this
     * class is for private use of the BinFile type.
     */
    struct TensorFileHeader {

      /**
       * Constructor
       */
      TensorFileHeader();

      /**
       * Destructor
       */
      virtual ~TensorFileHeader();

      /**
       * Gets the shape of each array in a blitz format
       */
      template<int D> void getShape (blitz::TinyVector<int,D>& res) const {
        for (int i=0; i<D; ++i) res[i] = m_shape[i];
      }

      /**
       * Sets the shape of each array
       */
      void setShape(size_t ndim, const size_t* shape) {
        m_n_dimensions = ndim;
        for(size_t i=0; i<ndim; ++i) m_shape[i] = shape[i];
      }

      /**
       * Gets the size along a particular dimension
       */
      size_t getSize(size_t dim_index) const;

      /**
       * Sets the size along a particular dimension
       */
      void setSize(const size_t dim_index, size_t val);

      /** 
       * Gets the offset of some array in the file
       */
      size_t getArrayIndex(size_t index) const;

      /**
       * Writes the header into an output stream
       */
      void write(std::ostream& str) const;

      /**
       * Reads the header from an input stream
       */
      void read(std::istream& str);

      /**
       * Gets number of elements in binary file
       */
      inline size_t getNElements() const {
        size_t tmp = 1;
        for(size_t i=0; i<m_n_dimensions; ++i) tmp *= m_shape[i];
        return tmp;
      }

      /**
       * Returns the number of dimensions in this binary file
       */
      inline size_t getNDim() const { return m_n_dimensions; }

      /**
       * Returns the shape in a N-element C-style array
       */
      inline const size_t* getShape() const { return m_shape; }
    
      /**
        * Checks if the header is valid
        */
      void header_ok();

      /**
        * Update the TensorSize value
        */
      void update();

      //representation
      Torch::database::TensorType m_tensor_type; ///< array element type 
      Torch::core::array::ElementType m_elem_type; ///< array element type 
      size_t m_n_samples; ///< total number of arrays in the file
      size_t m_n_dimensions; ///< the number of dimensions in each array
      size_t m_shape[Torch::core::array::N_MAX_DIMENSIONS_ARRAY]; ///< shape of data
      size_t m_tensor_size; ///< the number of dimensions in each array
    };

} } }

#endif /* TORCH_DATABASE_TENSORFILEHEADER_H */
