/**
 * @author <a href="mailto:andre.dos.anjos@gmail.com">Andre Anjos</a> 
 * @date Mon 21 Feb 13:51:53 2011 
 *
 * @brief Utilities to read and write .mat (Matlab) binary files
 */

#ifndef TORCH_DATABASE_MATUTILS_H 
#define TORCH_DATABASE_MATUTILS_H

#include <matio.h>
#include <blitz/array.h>

#include "core/array_common.h"
#include "database/InlinedArrayImpl.h"
#include "database/Exception.h"

namespace Torch { namespace database { namespace detail {

  /**
   * Returns the MAT_C_* enumeration for the given ElementType
   */
  enum matio_classes mio_class_type (Torch::core::array::ElementType i);

  /**
   * Returns the MAT_T_* enumeration for the given ElementType
   */
  enum matio_types mio_data_type (Torch::core::array::ElementType i);

  /**
   * Returns the ElementType given the matio MAT_T_* enum and a flag indicating
   * if the array is complex or not (also returned by matio at matvar_t)
   */
  Torch::core::array::ElementType torch_element_type (int mio_type, bool is_complex);

  //the next methods implement compile-time switching using templates
  template <int D> blitz::TinyVector<int,D> make_shape (const int* shape) {
    throw Torch::database::DimensionError(D, Torch::core::array::N_MAX_DIMENSIONS_ARRAY);
  }
  template <> blitz::TinyVector<int,1> make_shape<1> (const int* shape);
  template <> blitz::TinyVector<int,2> make_shape<2> (const int* shape);
  template <> blitz::TinyVector<int,3> make_shape<3> (const int* shape);
  template <> blitz::TinyVector<int,4> make_shape<4> (const int* shape);
 
  /**
   * Reads a variable on the (already opened) mat_t file. If you don't
   * specify the variable name, I'll just read the next one.
   */
  template <typename T, int D> InlinedArrayImpl read_array(mat_t* file, 
      const char* varname=0) {

    matvar_t* matvar = 0;
    if (varname) matvar = Mat_VarRead(file, const_cast<char*>(varname));
    else matvar = Mat_VarReadNext(file);

    blitz::Array<T,D> data(static_cast<T*>(matvar->data), 
        make_shape<D>(matvar->dims), blitz::duplicateData);
    Mat_VarFree(matvar);

    return InlinedArrayImpl(data); 
  }

  /**
   * Appends a single Array into the given matlab file and with a given name
   */
  template <typename T, int D>
    void write_array(mat_t* file, const char* varname, const InlinedArrayImpl& data) {

      Torch::core::array::ElementType eltype = data.getElementType();
      blitz::Array<T,D> bzdata = data.get<T,D>();
      if (!bzdata.isStorageContiguous()) bzdata.reference(bzdata.copy());

      //matio gets dimensions as integers
      int mio_dims[Torch::core::array::N_MAX_DIMENSIONS_ARRAY];
      for (size_t i=0; i<D; ++i) mio_dims[i] = data.getShape()[i];

      matvar_t* matvar = Mat_VarCreate(varname, 
          mio_class_type(eltype), mio_data_type(eltype), 
          D, mio_dims, static_cast<void*>(bzdata.data()), MEM_CONSERVE);

      Mat_VarWrite(file, matvar, 0);
      Mat_VarFree(matvar);
    }

  /**
   * Reads a complex variable on the (already opened) mat_t file. If you don't
   * specify the variable name, I'll just read the next one.
   */
  template <typename T, typename F, int D> InlinedArrayImpl read_complex_array
    (mat_t* file, const char* varname=0) {

    matvar_t* matvar = 0;
    if (varname) matvar = Mat_VarRead(file, const_cast<char*>(varname));
    else matvar = Mat_VarReadNext(file);

    //copies the pointers of interest.
    ComplexSplit mio_complex = *static_cast<ComplexSplit*>(matvar->data);
    F* real = static_cast<F*>(mio_complex.Re);
    F* imag = static_cast<F*>(mio_complex.Im);

    blitz::Array<T,D> data(make_shape<D>(matvar->dims));
    size_t n=0;
    for (typename blitz::Array<T,D>::iterator it=data.begin(); it!=data.end(); ++it, ++n) {
      (*it) = std::complex<F>(real[n], imag[n]);
    }
    Mat_VarFree(matvar);

    return InlinedArrayImpl(data); 
  }

  /**
   * Appends a single complex Array into the given matlab file and with a given
   * name
   */
  template <typename T, typename F, int D>
    void write_complex_array(mat_t* file, const char* varname, 
        const InlinedArrayImpl& data) {

      Torch::core::array::ElementType eltype = data.getElementType();
      blitz::Array<T,D> bzdata = data.get<T,D>();

      //matio accepts real/imaginary parts separated in a ComplexSplit struct.
      //The user must do the separation him/herself. 
 
      blitz::Array<F,D> bzre = blitz::real(bzdata);
      if (!bzre.isStorageContiguous()) bzre.reference(bzre.copy());
      blitz::Array<F,D> bzim = blitz::imag(bzdata);
      if (!bzim.isStorageContiguous()) bzim.reference(bzim.copy());

      ComplexSplit mio_complex = { 
        static_cast<void*>(bzre.data()),
        static_cast<void*>(bzim.data()) 
      };

      //matio gets dimensions as integers
      int mio_dims[Torch::core::array::N_MAX_DIMENSIONS_ARRAY];
      for (size_t i=0; i<D; ++i) mio_dims[i] = data.getShape()[i];

      matvar_t* matvar = Mat_VarCreate(varname, 
          mio_class_type(eltype), mio_data_type(eltype),
          D, mio_dims, 
          static_cast<void*>(&mio_complex), MEM_CONSERVE|MAT_F_COMPLEX);

      Mat_VarWrite(file, matvar, 0);
      Mat_VarFree(matvar);
    }

}}}

#endif /* TORCH_DATABASE_MATUTILS_H */
