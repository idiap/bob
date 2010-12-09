/**
 * @file src/core/core/Dataset2.h
 * @author <a href="mailto:Laurent.El-Shafey@idiap.ch">Laurent El Shafey</a>
 * @author <a href="mailto:andre.anjos@idiap.ch">Andre Anjos</a>
 *
 * @brief A torch representation of a Dataset
 */

#ifndef TORCH5SPRO_CORE_DATASET_H
#define TORCH5SPRO_CORE_DATASET_H

#include <boost/shared_ptr.hpp>
#include <boost/shared_array.hpp>
#include <blitz/array.h>
#include "core/logging.h"
#include "core/Exception.h"
#include <string>
#include <map>
#include <cstdlib> // required when using size_t type


namespace Torch {   
  /**
   * \ingroup libcore_api
   * @{
   *
   */
  namespace core {
  
    typedef enum ArrayType { t_unknown, t_bool, 
      t_int8, t_int16, t_int32, t_int64, 
      t_uint8, t_uint16, t_uint32, t_uint64, 
      t_float32, t_float64, t_float128,
      t_complex64, t_complex128, t_complex256 } ArrayType;

    typedef enum LoaderType { l_unknown, l_blitz, l_tensor, l_bindata } 
      LoaderType;

    
    // Declare the Arrayset for the reference to the parent Arrayset in the
    // Array class.
    class Arrayset;

    /**
     * @brief The array class for a dataset
     */
    class Array { //pure virtual
      public:
        /**
         * @brief Constructor
         */
        Array(const Arrayset& parent);
        /**
         * @brief Destructor
         */
        ~Array();
        
        /**
         * @brief Set the id of the Array
         */
        void setId(const size_t id) { m_id = id; }
        /**
         * @brief Set the flag indicating if this array is loaded from an 
         * external file.
         */
        void setIsLoaded(const bool is_loaded) { m_is_loaded = is_loaded; }
        /**
         * @brief Set the filename containing the data if any. An empty string
         * indicates that the data are stored in the XML file directly.
         */
        void setFilename(const std::string& filename)
          { m_filename.assign(filename); }
        /**
         * @brief Set the loader used to read the data from the external file 
         * if any.
         */
        void setLoader(const LoaderType loader) { m_loader = loader; }
        /**
         * @brief Set the data of the Array. Storage should have been allocated
         * with malloc, to make the deallocation easy? 
         */
        void setStorage(void* storage) { m_storage = storage; }

        /**
         * @brief Get the id of the Array
         */
        const size_t getId() const { return m_id; }
        /**
         * @brief Get the flag indicating if the array is loaded from an 
         * external file.
         */
        const bool getIsLoaded() const { return m_is_loaded; }
        /**
         * @brief Get the filename containing the data if any. An empty string
         * indicates that the data is stored in the XML file directly.
         */
        const std::string& getFilename() const { return m_filename; }
        /**
         * @brief Get the loader used to read the data from the external file 
         * if any.
         */
        const LoaderType getLoader() const {return m_loader; }

        //TODO: method to get the data???
        template<typename T, int D> void copy( blitz::Array<T,D>& output);

      private:
        template <typename T, typename U> void copyCast( T* out);

        const Arrayset& m_parent_arrayset;
        size_t m_id;
        bool m_is_loaded;
        std::string m_filename;
        LoaderType m_loader;
        void* m_storage;
    };



    /**
     * @brief The arrayset class for a dataset
     */
    class Arrayset { //pure virtual
      //
      //query/iterate over:
      //1. "Array"
      public:
        /**
         * @brief Constructor
         */
        Arrayset();
        /**
         * @brief Destructor
         */
        ~Arrayset();

        /**
         * @brief Add an Array to the Arrayset
         */
        // TODO: const argument or not?
        void addArray( boost::shared_ptr<Array> array);

        /**
         * @brief Set the id of the Arrayset
         */
        void setId(const size_t id) { m_id = id; }
        /**
         * @brief Set the number of dimensions of the arrays of this Arrayset
         */
        void setNDim(const size_t n_dim) { m_n_dim = n_dim; }
        /**
         * @brief Set the size of each dimension of the arrays of this 
         * Arrayset
         */
        void setShape(const size_t shape[]) { 
          for(size_t i=0; i<4; ++i)
            m_shape[i] = shape[i];
        }
        /**
         * @brief Set the number of elements in each array of this 
         * Arrayset
         */
        void setNElem(const size_t n_elem) {  m_n_elem = n_elem; } 
        /**
         * @brief Set the type of the elements contained in the the arrays of 
         * this Arrayset
         */
        void setArrayType(const ArrayType element_type) 
          { m_element_type = element_type; }
        /**
         * @brief Set the role of the Arrayset
         */
        void setRole(const std::string& role) {m_role.assign(role); } 
        /**
         * @brief Set the flag indicating if this arrayset is loaded from an 
         * external file.
         */
        void setIsLoaded(const bool is_loaded) { m_is_loaded = is_loaded; }
        /**
         * @brief Set the filename containing the data if any. An empty string
         * indicates that the data are stored in the XML file directly.
         */
        void setFilename(const std::string& filename)
          { m_filename.assign(filename); }
        /**
         * @brief Set the loader used to read the data from the external file 
         * if any.
         */
        void setLoader(const LoaderType loader) { m_loader = loader; }
        
        /**
         * @brief Get the id of the Arrayset
         */
        size_t getId() const { return m_id; }
        /**
         * @brief Get the number of dimensions of the arrays of this Arrayset
         */
        size_t getNDim() const { return m_n_dim; }
        /**
         * @brief Get the size of each dimension of the arrays of this 
         * Arrayset
         */
        const size_t* getShape() const { return m_shape; }
        /**
         * @brief Get the number of elements in each array of this 
         * Arrayset
         */
        const size_t getNElem() const { return m_n_elem; } 
        /**
         * @brief Get the type of the elements contained in the the arrays of 
         * this Arrayset
         */
        ArrayType getArrayType() const { return m_element_type; }
        /**
         * @brief Get the role of this Arrayset
         */
        const std::string& getRole() const { return m_role; }
        /**
         * @brief Get the flag indicating if the arrayset is loaded from an 
         * external file.
         */
        bool getIsLoaded() const { return m_is_loaded; }
        /**
         * @brief Get the filename containing the data if any. An empty string
         * indicates that the data is stored in the XML file directly.
         */
        const std::string& getFilename() const { return m_filename; }
        /**
         * @brief Get the loader used to read the data from the external file 
         * if any.
         */
        LoaderType getLoader() const {return m_loader; }


        /**
         * @brief const_iterator over the Arrays of the Arrayset
         */
        typedef std::map<size_t, boost::shared_ptr<Array> >::const_iterator 
          const_iterator;
        /**
         * @brief Return a const_iterator pointing at the first Array of the 
         * Arrayset
         */
        const_iterator begin() const { return m_array.begin(); }
        /**
         * @brief Return a const_iterator pointing at the last Array of the 
         * Arrayset
         */
        const_iterator end() const { return m_array.end(); }

        /**
         * @brief iterator over the Arrays of the Arrayset
         */
        typedef std::map<size_t, boost::shared_ptr<Array> >::iterator 
          iterator;
        /**
         * @brief Return an iterator pointing at the first Array of the 
         * Arrayset
         */
        iterator begin() { return m_array.begin(); }
        /**
         * @brief Return an iterator pointing at the last Array of the 
         * Arrayset
         */
        iterator end() { return m_array.end(); }

        /**
         * @brief Update the blitz array with the content of the array 
         * of the provided id.
         */
        template<typename T, int D> 
        void at(size_t id, blitz::Array<T,D>& output);
        // blitz::Array<float, 2> myarray;
        // arrayset->at(3, myarray);
        //
      private:
        size_t m_id;

        size_t m_n_dim;
        size_t m_shape[4];
        size_t m_n_elem;
        ArrayType m_element_type;
        
        std::string m_role;
        bool m_is_loaded;
        std::string m_filename;
        LoaderType m_loader;

        std::map<size_t, boost::shared_ptr<Array> > m_array;
    };


    /**
     * @brief The relation class for a dataset
     */
    class Relation { //pure virtual
      // TODO
    };

    /**
     * @brief The rule class for a dataset
     */
    class Rule { //pure virtual
      // TODO
    };

    /**
     * @brief The relationset class for a dataset
     */
    class Relationset { //pure virtual
      // TODO
    };


    /**
     * @brief The main dataset class
    */
    class Dataset { //pure virtual
      public:
        /**
         * @brief Constructor
         */
        Dataset();
        /**
         * @brief Destructor
         */
        ~Dataset();

        /**
         * @brief Add an Arrayset to the Dataset
         */
        // TODO: const argument or not?
        void addArrayset( boost::shared_ptr<Arrayset> arrayset);

        /**
         * @brief const_iterator over the Arraysets of the Dataset
         */
        typedef std::map<size_t, boost::shared_ptr<Arrayset> >::const_iterator 
          const_iterator;
        /**
         * @brief Return a const_iterator pointing at the first Arrayset of 
         * the Arrayset
         */
        const_iterator begin() const { return m_arrayset.begin(); }
        /**
         * @brief Return a const_iterator pointing at the last Arrayset of 
         * the Arrayset
         */
        const_iterator end() const { return m_arrayset.end(); }

        /**
         * @brief iterator over the Arraysets of the Dataset
         */
        typedef std::map<size_t, boost::shared_ptr<Arrayset> >::iterator iterator;
        /**
         * @brief Return an iterator pointing at the first Arrayset of 
         * the Arrayset
         */
        iterator begin() { return m_arrayset.begin(); }
        /**
         * @brief Return an iterator pointing at the first Arrayset of 
         * the Arrayset
         */
        iterator end() { return m_arrayset.end(); }
   
        /**
         * @brief Return the Arrayset of the given id 
         */
        const boost::shared_ptr<Arrayset> at( const size_t id ) const;

      private:    
        std::map<size_t, boost::shared_ptr<Arrayset> > m_arrayset;
        std::map<size_t, boost::shared_ptr<Relationset> > m_relationset;
    };


    /********************** TEMPLATE FUNCTION DEFINITIONS ***************/
    template <typename T, typename U> 
    void Array::copyCast( T* out) {
      size_t n_elem = m_parent_arrayset.getNElem();
      for( size_t i=0; i<n_elem; ++i) {
        U* u_storage = reinterpret_cast<U*>(m_storage);
        out[i] = *reinterpret_cast<T*>(&u_storage[i]);
      }
    }

    template <typename T, int D> 
    void Array::copy( blitz::Array<T,D>& output) 
    {
      if( D != m_parent_arrayset.getNDim() ) {
        std::cout << "D=" << D << " -- ParseXML: D=" <<
           m_parent_arrayset.getNDim() << std::endl;
        error << "Cannot copy the data in a blitz array with a different " <<
          "number of dimensions." << std::endl;
        throw Exception();
      }

      if( output.numElements() != m_parent_arrayset.getNElem() ) {
        error << "Cannot copy the data in a blitz array with a different " <<
          "number of elements." << std::endl;
        throw Exception();
      }

      // TODO: check number of elements in each dimensions?
      T* out_data = output.data();
      switch(m_parent_arrayset.getArrayType()) {
        case t_bool:
          //output = blitz::Array<T,D>(reinterpret_cast<T>(m_storage), (shape), blitz::duplicateData);
          copyCast<T,bool>(out_data); break;
        case t_int8:
          copyCast<T,int8_t>(out_data); break;
        case t_int16:
          copyCast<T,int16_t>(out_data); break;
        case t_int32:
          copyCast<T,int32_t>(out_data); break;
        case t_int64:
          copyCast<T,int64_t>(out_data); break;
        case t_uint8:
          copyCast<T,uint8_t>(out_data); break;
        case t_uint16:
          copyCast<T,uint16_t>(out_data); break;
        case t_uint32:
          copyCast<T,uint32_t>(out_data); break;
        case t_uint64:
          copyCast<T,uint64_t>(out_data); break;
        case t_float32:
          copyCast<T,float>(out_data); break;
        case t_float64:
          copyCast<T,double>(out_data); break;
        case t_float128:
          copyCast<T,long double>(out_data); break;
        case t_complex64:
          copyCast<T,std::complex<float> >(out_data); break;
        case t_complex128:
          copyCast<T,std::complex<double> >(out_data); break;
        case t_complex256:
          copyCast<T,std::complex<long double> >(out_data); break;
        default:
          break;
      }
    }


    template<typename T, int D> void 
    Arrayset::at(size_t id, blitz::Array<T,D>& output) {
      boost::shared_ptr<Array> x = (m_array.find(id))->second;
      x->copy(output);
    }



  }
  /**
   * @}
   */
}

#endif /* TORCH5SPRO_CORE_DATASET_H */

