/**
 * @author <a href="mailto:andre.dos.anjos@gmail.com">Andre Anjos</a> 
 * @date Mon 21 Feb 13:51:53 2011 
 *
 * @brief Utilities to read and write .mat (Matlab) binary files
 */

#ifndef TORCH_IO_MATUTILS_H 
#define TORCH_IO_MATUTILS_H

#include "core/array_type.h"
#include <blitz/array.h>

#include <matio.h>
#include <boost/format.hpp>
#include <boost/shared_ptr.hpp>
#include <map>
#include <string>

#include "core/array_check.h"
#include "core/array_copy.h"
#include "io/InlinedArrayImpl.h"
#include "io/InlinedArraysetImpl.h"
#include "io/Exception.h"

namespace Torch { namespace io { namespace detail {

  /**
   * This structure holds information about array typing in a single bound
   * place.
   * 
   */
  class ArrayTypeInfo {

    public:
      //what defines an array in terms of its typing...
      Torch::core::array::ElementType eltype;
      size_t ndim;
      size_t shape[Torch::core::array::N_MAX_DIMENSIONS_ARRAY];

      /**
       * Compares two ArrayTypeInfo objects for equality
       */
      bool operator== (const ArrayTypeInfo& other) const;

      /**
       * Compares two ArrayTypeInfo objects for inequality
       */
      inline bool operator!= (const ArrayTypeInfo& other) const {
        return ! (*this == other);
      }

  };

  /**
   * This method will create a new boost::shared_ptr to mat_t that knows how to
   * delete itself
   */
  boost::shared_ptr<mat_t> make_matfile(const std::string& filename, int flags);

  /**
   * This method will create a new boost::shared_ptr to matvar_t that knows how
   * to delete itself
   *
   * You pass it the file and the variable name to be read out or a combination
   * of parameters required to build a new matvar_t from scratch (see matio API
   * for details).
   */
  boost::shared_ptr<matvar_t> make_matvar(boost::shared_ptr<mat_t>& file);
  boost::shared_ptr<matvar_t> make_matvar(boost::shared_ptr<mat_t>& file,
      const std::string& varname);

  /**
   * This is essentially like make_matvar(), but uses VarReadNextInfo() instead
   * of VarReadNext(), so it does not load the data, but it is faster.
   */
  boost::shared_ptr<matvar_t> make_matvar_info(boost::shared_ptr<mat_t>& file);

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
  Torch::core::array::ElementType torch_element_type 
    (int mio_type, bool is_complex);

  /**
   * The next two variants will create matvar's that auto-destruct based on the
   * contents of an InlinedArrayImpl (complex and non-complex forms). They
   * return pairs which include the blitz array that is extracted from the
   * InlinedArrayImpl (and copied), since that holds the memory to which the
   * matvar_t object is pointing to.
   */
  template <typename T, int D> std::pair<blitz::Array<T,D>, 
           boost::shared_ptr<matvar_t> > make_matpair
    (const std::string& varname, const InlinedArrayImpl& data) {

      Torch::core::array::ElementType eltype = data.getElementType();
      blitz::Array<T,D> c_style = data.get<T,D>();

      //transform the input data into a fortran-style array as Matlab requires
      //fortran order for storage.
      blitz::Array<T,D> bzdata(c_style.shape(), blitz::FortranArray<D>());
      bzdata = c_style; //copy

      //matio gets dimensions as integers
      int mio_dims[Torch::core::array::N_MAX_DIMENSIONS_ARRAY];
      for (size_t i=0; i<D; ++i) mio_dims[i] = data.getShape()[i];

      return std::make_pair(bzdata,
          boost::shared_ptr<matvar_t>(Mat_VarCreate(varname.c_str(), 
              mio_class_type(eltype), mio_data_type(eltype), 
              D, mio_dims, static_cast<void*>(bzdata.data()), MEM_CONSERVE),
            std::ptr_fun(Mat_VarFree)));
    }

  template <typename T, typename F, int D> 
    std::pair<std::pair<blitz::Array<F,D>, blitz::Array<F,D> >,
    boost::shared_ptr<matvar_t> > make_matpair
      (const std::string& varname, const InlinedArrayImpl& data) {

      Torch::core::array::ElementType eltype = data.getElementType();
      blitz::Array<T,D> bzdata = data.get<T,D>();
      
      //matio accepts real/imaginary parts separated in a ComplexSplit struct.
      //The user must do the separation him/herself. 

      //transform the input data into a fortran-style array as Matlab requires
      //fortran order for storage.
      blitz::Array<F,D> bzre(bzdata.shape(), blitz::FortranArray<D>());
      bzre = Torch::core::array::ccopy(blitz::real(bzdata)); //makes contiguous
      blitz::Array<F,D> bzim(bzdata.shape(), blitz::FortranArray<D>());
      bzim = Torch::core::array::ccopy(blitz::imag(bzdata)); //makes contiguous

      ComplexSplit mio_complex = { 
        static_cast<void*>(bzre.data()),
        static_cast<void*>(bzim.data()) 
      };

      //matio gets dimensions as integers
      int mio_dims[Torch::core::array::N_MAX_DIMENSIONS_ARRAY];
      for (size_t i=0; i<D; ++i) mio_dims[i] = data.getShape()[i];

      return std::make_pair(std::make_pair(bzre, bzim), 
          boost::shared_ptr<matvar_t>(Mat_VarCreate(varname.c_str(), 
              mio_class_type(eltype), mio_data_type(eltype),
              D, mio_dims, static_cast<void*>(&mio_complex),
              MEM_CONSERVE|MAT_F_COMPLEX), std::ptr_fun(Mat_VarFree)));
    }


  //the next methods implement compile-time switching using templates
  template <int D> blitz::TinyVector<int,D> make_shape (const int* shape) {
    throw Torch::io::DimensionError(D, 
        Torch::core::array::N_MAX_DIMENSIONS_ARRAY);
  }
  template <> blitz::TinyVector<int,1> make_shape<1> (const int* shape);
  template <> blitz::TinyVector<int,2> make_shape<2> (const int* shape);
  template <> blitz::TinyVector<int,3> make_shape<3> (const int* shape);
  template <> blitz::TinyVector<int,4> make_shape<4> (const int* shape);

  /**
   * Assignes a single matvar variable to a blitz::Array
   */
  template <typename T, int D> InlinedArrayImpl assign_array
    (boost::shared_ptr<matvar_t> matvar) {
    //note: Matlab saves data in fortran style (column major order)
    blitz::Array<T,D> f_data(static_cast<T*>(matvar->data), 
        make_shape<D>(matvar->dims), blitz::duplicateData, 
        blitz::FortranArray<D>());
    blitz::Array<T,D> data(f_data.shape());
    data = f_data; //copies data to c-style array
    return InlinedArrayImpl(data); 
  }
 
  /**
   * Reads a variable on the (already opened) mat_t file. If you don't
   * specify the variable name, I'll just read the next one.
   */
  template <typename T, int D> InlinedArrayImpl read_array
    (boost::shared_ptr<mat_t> file, const std::string& varname="") {

    boost::shared_ptr<matvar_t> matvar;
    if (varname.size()) matvar = make_matvar(file, varname);
    else matvar = make_matvar(file);
    if (!matvar) throw Torch::io::Uninitialized();
    InlinedArrayImpl retval = assign_array<T,D>(matvar);
    return retval;

  }

  /**
   * Appends a single Array into the given matlab file and with a given name
   */
  template <typename T, int D>
    void write_array(boost::shared_ptr<mat_t> file, const std::string& varname, 
        const InlinedArrayImpl& data) {

      std::pair<blitz::Array<T,D>, boost::shared_ptr<matvar_t> > matpair = 
        make_matpair<T,D>(varname, data);
      Mat_VarWrite(file.get(), matpair.second.get(), 0);

    }

  template <typename T, typename F, int D> InlinedArrayImpl assign_complex_array
    (boost::shared_ptr<matvar_t> matvar) {
      //copies the pointers of interest.
      ComplexSplit mio_complex = *static_cast<ComplexSplit*>(matvar->data);
      F* real = static_cast<F*>(mio_complex.Re);
      F* imag = static_cast<F*>(mio_complex.Im);

      //note: Matlab saves data in fortran style (column major order)
      blitz::Array<T,D> f_data(make_shape<D>(matvar->dims),
          blitz::FortranArray<D>());
      size_t n=0;
      for (typename blitz::Array<T,D>::iterator 
          it=f_data.begin(); it!=f_data.end(); ++it, ++n) {
        (*it) = std::complex<F>(real[n], imag[n]);
      }
      blitz::Array<T,D> data(f_data.shape());
      data = f_data; //copies data to c-style array
      return InlinedArrayImpl(data); 
    }

  /**
   * Reads a complex variable on the (already opened) mat_t file. If you don't
   * specify the variable name, I'll just read the next one.
   */
  template <typename T, typename F, int D> InlinedArrayImpl read_complex_array
    (boost::shared_ptr<mat_t> file, const std::string& varname="") {
      boost::shared_ptr<matvar_t> matvar;
      if (varname.size()) matvar = make_matvar(file, varname);
      else matvar = make_matvar(file);
      if (!matvar) throw Torch::io::Uninitialized();
      InlinedArrayImpl retval = assign_complex_array<T,F,D>(matvar);
      return retval;
  }

  /**
   * Appends a single complex Array into the given matlab file and with a given
   * name
   */
  template <typename T, typename F, int D>
    void write_complex_array(boost::shared_ptr<mat_t> file, 
        const std::string& varname, const InlinedArrayImpl& data) {

      std::pair<std::pair<blitz::Array<F,D>, blitz::Array<F,D> >,
      boost::shared_ptr<matvar_t> > matpair = make_matpair<T,F,D>(varname, data); 
      Mat_VarWrite(file.get(), matpair.second.get(), 0);


    }

  /**
   * Reads all variables on the (already opened) mat_t file.
   */
  template <typename T, int D> InlinedArraysetImpl read_arrayset
    (boost::shared_ptr<mat_t> file) {
      InlinedArraysetImpl retval;
      boost::shared_ptr<matvar_t> matvar;
      while ((matvar = make_matvar(file))) {
        retval.add(assign_array<T,D>(matvar));
      }
      return retval;
    }

  /**
   * Writes all arrays in the given arrayset
   */
  template <typename T, int D> void write_arrayset
    (boost::shared_ptr<mat_t> file, boost::format& fmt_varname,
     const InlinedArraysetImpl& data) {
      for (size_t i=0; i<data.size(); ++i) {
        fmt_varname % (i);
        write_array<T,D>(file, fmt_varname.str(), data[i].get());
      }
  }

  /**
   * Reads all complex variables on the (already opened) mat_t file.
   */
  template <typename T, typename F, int D> InlinedArraysetImpl 
    read_complex_arrayset (boost::shared_ptr<mat_t> file) {
      InlinedArraysetImpl retval;
      boost::shared_ptr<matvar_t> matvar;
      while ((matvar = make_matvar(file))) {
        retval.add(assign_complex_array<T,F,D>(matvar));
      }
      return retval;
  }

  /**
   * Writes all complex arrays in the given arrayset
   */
  template <typename T, typename F, int D> void write_complex_arrayset
    (boost::shared_ptr<mat_t> file, boost::format& fmt_varname,
     const InlinedArraysetImpl& data) {
      for (size_t i=0; i<data.size(); ++i) {
        fmt_varname % (i);
        write_complex_array<T,F,D>(file, fmt_varname.str(), data[i].get());
      }
  }

  /**
   * Given a matvar_t object, returns our equivalent ArrayTypeInfo struct.
   */
  void get_info(boost::shared_ptr<const matvar_t> matvar, 
      ArrayTypeInfo& info);

  /**
   * Retrieves information about the first variable with a certain name
   * (array_%d) that exists in a .mat file (if it exists)
   */
  void get_info_first(const std::string& filename, ArrayTypeInfo& info);

  /**
   * Retrieves information about all variables with a certain name (array_%d)
   * that exist in a .mat file
   */
  boost::shared_ptr<std::map<size_t, 
    std::pair<std::string, ArrayTypeInfo> > > 
      list_variables(const std::string& filename);

}}}

#endif /* TORCH_IO_MATUTILS_H */
