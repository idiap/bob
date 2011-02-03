/**
 * @file src/cxx/core/core/Arrayset.h
 * @author <a href="mailto:Laurent.El-Shafey@idiap.ch">Laurent El Shafey</a>
 *
 * @brief A torch representation of an Array for a Dataset.
 */

#ifndef TORCH5SPRO_CORE_ARRAYSET_H
#define TORCH5SPRO_CORE_ARRAYSET_H 1

#include <boost/shared_ptr.hpp>
#include <boost/shared_array.hpp>
#include <blitz/array.h>
#include "core/logging.h"
#include "core/Exception.h"
#include "core/StaticComplexCast.h"
#include "core/dataset_common.h"

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
    
    // Declare Arrayset for the reference to the parent Arrayset in the
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
         * @brief Set the flag indicating if this array is loaded.
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
        const LoaderType getLoader() const { return m_loader; }
        /**
         * @brief Get a pointer to the storage area containing the data
         */
        const void* getStorage() const { 
          if(!getIsLoaded()) {
            ;//TODO:load
            Array* a=const_cast<Array*>(this);
            a->setIsLoaded(true);
          }          
          return m_storage; 
        }
        /**
         * @brief Get the parent arrayset of this array
         */
        const Arrayset& getParentArrayset() const { return m_parent_arrayset; }

        /**
         * @brief Adapt the size of each dimension of the passed blitz array
         * to the ones of the underlying array and copy the data in it.
         */
        template<typename T, int D> 
        void copy( blitz::Array<T,D>& output) const;

/*
        template<typename T, int D> struct do_refer
        {
          blitz::Array<T,D> static apply(Array&) {
            error << "Unsupported blitz array type " << std::endl;
            throw TypeError();
          }
        };

        template<int D> struct do_refer<bool,D> {
          blitz::Array<bool,D> static apply(Array& arr);
        };
        friend struct do_refer<bool,1>;
        friend struct do_refer<bool,2>;
        friend struct do_refer<bool,3>;
        friend struct do_refer<bool,4>;

        template<int D> struct do_refer<int8_t,D> {
          blitz::Array<int8_t,D> static apply(Array& arr);
        };
        friend struct do_refer<int8_t,1>;
        friend struct do_refer<int8_t,2>;
        friend struct do_refer<int8_t,3>;
        friend struct do_refer<int8_t,4>;

        template<int D> struct do_refer<int16_t,D> {
          blitz::Array<int16_t,D> static apply(Array& arr);
        };
        friend struct do_refer<int16_t,1>;
        friend struct do_refer<int16_t,2>;
        friend struct do_refer<int16_t,3>;
        friend struct do_refer<int16_t,4>;

        template<int D> struct do_refer<int32_t,D> {
          blitz::Array<int32_t,D> static apply(Array& arr);
        };
        friend struct do_refer<int32_t,1>;
        friend struct do_refer<int32_t,2>;
        friend struct do_refer<int32_t,3>;
        friend struct do_refer<int32_t,4>;

        template<int D> struct do_refer<int64_t,D> {
          blitz::Array<int64_t,D> static apply(Array& arr);
        };
        friend struct do_refer<int64_t,1>;
        friend struct do_refer<int64_t,2>;
        friend struct do_refer<int64_t,3>;
        friend struct do_refer<int64_t,4>;

        template<int D> struct do_refer<uint8_t,D> {
          blitz::Array<uint8_t,D> static apply(Array& arr);
        };
        friend struct do_refer<uint8_t,1>;
        friend struct do_refer<uint8_t,2>;
        friend struct do_refer<uint8_t,3>;
        friend struct do_refer<uint8_t,4>;

        template<int D> struct do_refer<uint16_t,D> {
          blitz::Array<uint16_t,D> static apply(Array& arr);
        };
        friend struct do_refer<uint16_t,1>;
        friend struct do_refer<uint16_t,2>;
        friend struct do_refer<uint16_t,3>;
        friend struct do_refer<uint16_t,4>;

        template<int D> struct do_refer<uint32_t,D> {
          blitz::Array<uint32_t,D> static apply(Array& arr);
        };
        friend struct do_refer<uint32_t,1>;
        friend struct do_refer<uint32_t,2>;
        friend struct do_refer<uint32_t,3>;
        friend struct do_refer<uint32_t,4>;

        template<int D> struct do_refer<uint64_t,D> {
          blitz::Array<uint64_t,D> static apply(Array& arr);
        };
        friend struct do_refer<uint64_t,1>;
        friend struct do_refer<uint64_t,2>;
        friend struct do_refer<uint64_t,3>;
        friend struct do_refer<uint64_t,4>;

        template<int D> struct do_refer<float,D> {
          blitz::Array<float,D> static apply(Array& arr);
        };
        friend struct do_refer<float,1>;
        friend struct do_refer<float,2>;
        friend struct do_refer<float,3>;
        friend struct do_refer<float,4>;

        template<int D> struct do_refer<double,D> {
          blitz::Array<double,D> static apply(Array& arr);
        };
        friend struct do_refer<double,1>;
        friend struct do_refer<double,2>;
        friend struct do_refer<double,3>;
        friend struct do_refer<double,4>;

        template<int D> struct do_refer<long double,D> {
          blitz::Array<long double,D> static apply(Array& arr);
        };
        friend struct do_refer<long double,1>;
        friend struct do_refer<long double,2>;
        friend struct do_refer<long double,3>;
        friend struct do_refer<long double,4>;

        template<int D> struct do_refer<std::complex<float>,D> {
          blitz::Array<std::complex<float>,D> static apply(Array& arr);
        };
        friend struct do_refer<std::complex<float>,1>;
        friend struct do_refer<std::complex<float>,2>;
        friend struct do_refer<std::complex<float>,3>;
        friend struct do_refer<std::complex<float>,4>;

        template<int D> struct do_refer<std::complex<double>,D> {
          blitz::Array<std::complex<double>,D> static apply(Array& arr);
        };
        friend struct do_refer<std::complex<double>,1>;
        friend struct do_refer<std::complex<double>,2>;
        friend struct do_refer<std::complex<double>,3>;
        friend struct do_refer<std::complex<double>,4>;

        template<int D> struct do_refer<std::complex<long double>,D> {
          blitz::Array<std::complex<long double>,D> static apply(Array& arr);
        };
        friend struct do_refer<std::complex<long double>,1>;
        friend struct do_refer<std::complex<long double>,2>;
        friend struct do_refer<std::complex<long double>,3>;
        friend struct do_refer<std::complex<long double>,4>;
*/
        /**
         * @brief Adapt the size of each dimension of the passed blitz array
         * to the ones of the underlying array and refer to the data in it.
         * @warning Updating the content of the blitz array will update the
         * content of the corresponding array in the dataset.
         */
        template<typename T, int D> blitz::Array<T,D> refer() {
          error << "Unsupported blitz array type " << std::endl;
          throw TypeError();
          //return do_refer<T,D>::apply(*this);
        }


        /************** Partial specialization declaration *************/
/*        template<int D> blitz::Array<bool,D> refer( );
        template<int D> blitz::Array<int8_t,D> refer( );
        template<int D> blitz::Array<int16_t,D> refer( );
        template<int D> blitz::Array<int32_t,D> refer( );
        template<int D> blitz::Array<int64_t,D> refer( );
        template<int D> blitz::Array<uint8_t,D> refer( );
        template<int D> blitz::Array<uint16_t,D> refer( );
        template<int D> blitz::Array<uint32_t,D> refer( );
        template<int D> blitz::Array<uint64_t,D> refer( );
        template<int D> blitz::Array<float,D> refer( );
        template<int D> blitz::Array<double,D> refer( );
        template<int D> blitz::Array<long double,D> refer( );
        template<int D> blitz::Array<std::complex<float>,D> refer( );
        template<int D> blitz::Array<std::complex<double>,D> refer( );
        template<int D> blitz::Array<std::complex<long double>,D> refer( );
*/
        template <typename T, typename U> void copyCast( U* out) const;

      private:
        template <int D> void referCheck( ) const;

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
    class Arrayset {
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
        void append( boost::shared_ptr<Array> array);
        /**
         * @brief Add a blitz array to the Arrayset
         */
        template <typename T, int D>
        void append( const blitz::Array<T,D>& bl);
        /**
         * @brief Remove an Array with a given id from the Arrayset
         */
        void remove( const size_t id) {
          std::map<size_t, boost::shared_ptr<Array> >::iterator it =
            m_array.find(id);
          if(it!=m_array.end())
            m_array.erase(it);
          else
            throw NonExistingElement();
        }

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
          for(size_t i=0; i<array::N_MAX_DIMENSIONS_ARRAY; ++i)
            m_shape[i] = shape[i];
        }
        /**
         * @brief Update the shape of the Array with the one given in the
         * blitz TinyVector.
         */
        template<int D> 
        void setShape( const blitz::TinyVector<int,D>& shape ) {
          m_n_dim = D;
          for( int i=0; i<array::N_MAX_DIMENSIONS_ARRAY; ++i)
            m_shape[i] = ( i<D ? shape(i) : 0);
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
        void setElementType(const array::ElementType element_type) 
          { m_element_type = element_type; }
        /**
         * @brief Set the role of the Arrayset
         */
        void setRole(const std::string& role) { m_role.assign(role); } 
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
         * @brief Update the given blitz array with the content of the array
         * of the provided id.
         */
        template<int D> void getShape( blitz::TinyVector<int,D>& res ) const {
          if( D!=m_n_dim )
            throw NDimensionError();
          for( int i=0; i<D; ++i)
            res[i] = m_shape[i];
        }
        /**
         * @brief Get the number of elements in each array of this 
         * Arrayset
         */
        size_t getNElem() const { return m_n_elem; } 
        /**
         * @brief Get the type of the elements contained in the the arrays of 
         * this Arrayset
         */
        array::ElementType getElementType() const { return m_element_type; }
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
        LoaderType getLoader() const { return m_loader; }
        /**
         * @brief Get the number of arrays in this Arrayset
         */
        size_t getNArrays() const { return m_array.size(); }


        /**
         * @brief const_iterator over the Arrays of the Arrayset
         */
        typedef std::map<size_t, boost::shared_ptr<Array> >::const_iterator 
          const_iterator;
        /**
         * @brief Return a const_iterator pointing at the first Array of the 
         * Arrayset
         */
        const_iterator begin() const { 
          if(!getIsLoaded()) {
            ;//TODO:load
            Arrayset* a=const_cast<Arrayset*>(this);
            a->setIsLoaded(true);
          }          
          return m_array.begin(); 
        }
        /**
         * @brief Return a const_iterator pointing at the last Array of the 
         * Arrayset
         */
        const_iterator end() const { 
          if(!getIsLoaded()) {
            ;//TODO:load
            Arrayset* a=const_cast<Arrayset*>(this);
            a->setIsLoaded(true);
          }          
          return m_array.end(); 
        }

        /**
         * @brief iterator over the Arrays of the Arrayset
         */
        typedef std::map<size_t, boost::shared_ptr<Array> >::iterator 
          iterator;
        /**
         * @brief Return an iterator pointing at the first Array of the 
         * Arrayset
         */
        iterator begin() { 
          if(!getIsLoaded()) {
            ;//TODO:load
            setIsLoaded(true);
          }          
          return m_array.begin(); 
        }
        /**
         * @brief Return an iterator pointing at the last Array of the 
         * Arrayset
         */
        iterator end() { 
          if(!getIsLoaded()) {
            ;//TODO:load
            setIsLoaded(true);
          }          
          return m_array.end(); 
        }

        /**
         * @brief Return the array of the given id
         * @warning Please note that if you use that method, scope matters,
         * because the dataset owns the arrays.
         */
        const Array& operator[](size_t id) const;
        /**
         * @brief Return a smart pointer to the array of the given id
         */
        boost::shared_ptr<const Array> getArray(size_t id) const;
        boost::shared_ptr<Array> getArray(size_t id);

        /**
         * @brief Update the blitz array with the content of the array 
         * of the provided id.
         */
        template<typename T, int D> 
        void at(size_t id, blitz::Array<T,D>& output) const;


      private:
        /**
         * @brief Check that the blitz array has a compatible number of 
         * dimensions with this Arrayset
         */
        template <int D> void appendCheck() const;

        size_t m_id;

        size_t m_n_dim;
        size_t m_shape[array::N_MAX_DIMENSIONS_ARRAY];
        size_t m_n_elem;
        array::ElementType m_element_type;
        
        std::string m_role;
        bool m_is_loaded;
        std::string m_filename;
        LoaderType m_loader;

        std::map<size_t, boost::shared_ptr<Array> > m_array;
    };


    /********************** TEMPLATE FUNCTION DEFINITIONS ***************/
    template <typename T, typename U> 
    void Array::copyCast( U* out ) const {
      size_t n_elem = m_parent_arrayset.getNElem();
      for( size_t i=0; i<n_elem; ++i) {
        T* t_storage = reinterpret_cast<T*>(m_storage);
        static_complex_cast( t_storage[i], out[i] );
      }
    }

    template <typename T, int D> 
    void Array::copy( blitz::Array<T,D>& output) const 
    {
      if( D != m_parent_arrayset.getNDim() ) {
        TDEBUG3("D=" << D << " -- ParseXML: D=" << 
          m_parent_arrayset.getNDim());
        error << "Cannot copy the data in a blitz array with a different " <<
          "number of dimensions." << std::endl;
        throw NDimensionError();
      }

      // Reshape each dimensions with the correct size
      blitz::TinyVector<int,D> shape;
      m_parent_arrayset.getShape(shape);
      output.resize(shape);

      // Load the data if required
      if(!getIsLoaded()) {
        ;//TODO:load 
        Array* a=const_cast<Array*>(this);
        a->setIsLoaded(true);
      }

      T* out_data;
      if( output.isStorageContiguous() )
        out_data = output.data();
      else
        out_data = output.copy().data();
      switch(m_parent_arrayset.getElementType()) {
        case array::t_bool:
          copyCast<bool,T>(out_data); break;
        case array::t_int8:
          copyCast<int8_t,T>(out_data); break;
        case array::t_int16:
          copyCast<int16_t,T>(out_data); break;
        case array::t_int32:
          copyCast<int32_t,T>(out_data); break;
        case array::t_int64:
          copyCast<int64_t,T>(out_data); break;
        case array::t_uint8:
          copyCast<uint8_t,T>(out_data); break;
        case array::t_uint16:
          copyCast<uint16_t,T>(out_data); break;
        case array::t_uint32:
          copyCast<uint32_t,T>(out_data); break;
        case array::t_uint64:
          copyCast<uint64_t,T>(out_data); break;
        case array::t_float32:
          copyCast<float,T>(out_data); break;
        case array::t_float64:
          copyCast<double,T>(out_data); break;
        case array::t_float128:
          copyCast<long double,T>(out_data); break;
        case array::t_complex64:
          copyCast<std::complex<float>,T>(out_data); break;
        case array::t_complex128:
          copyCast<std::complex<double>,T>(out_data); break;
        case array::t_complex256:
          copyCast<std::complex<long double>,T>(out_data); break;
        default:
          break;
      }
    }

    template <int D> void Array::referCheck( ) const
    {
      // Load the data if required
      if(!getIsLoaded()) {
        ;//TODO:load 
        Array* a=const_cast<Array*>(this);
        a->setIsLoaded(true);
      }

      if( D != m_parent_arrayset.getNDim() ) {
        TDEBUG3("D=" << D << " -- ParseXML: D=" <<
           m_parent_arrayset.getNDim());
        error << "Cannot refer to the data in a blitz array with a " <<
          "different number of dimensions." << std::endl;
        throw NDimensionError();
      }
    }


#define REFER_DECL(T,D) template<> \
   blitz::Array<T,D> Array::refer(); \

        REFER_DECL(bool,1)
        REFER_DECL(bool,2)
        REFER_DECL(bool,3)
        REFER_DECL(bool,4)
        REFER_DECL(int8_t,1)
        REFER_DECL(int8_t,2)
        REFER_DECL(int8_t,3)
        REFER_DECL(int8_t,4)
        REFER_DECL(int16_t,1)
        REFER_DECL(int16_t,2)
        REFER_DECL(int16_t,3)
        REFER_DECL(int16_t,4)
        REFER_DECL(int32_t,1)
        REFER_DECL(int32_t,2)
        REFER_DECL(int32_t,3)
        REFER_DECL(int32_t,4)
        REFER_DECL(int64_t,1)
        REFER_DECL(int64_t,2)
        REFER_DECL(int64_t,3)
        REFER_DECL(int64_t,4)
        REFER_DECL(uint8_t,1)
        REFER_DECL(uint8_t,2)
        REFER_DECL(uint8_t,3)
        REFER_DECL(uint8_t,4)
        REFER_DECL(uint16_t,1)
        REFER_DECL(uint16_t,2)
        REFER_DECL(uint16_t,3)
        REFER_DECL(uint16_t,4)
        REFER_DECL(uint32_t,1)
        REFER_DECL(uint32_t,2)
        REFER_DECL(uint32_t,3)
        REFER_DECL(uint32_t,4)
        REFER_DECL(uint64_t,1)
        REFER_DECL(uint64_t,2)
        REFER_DECL(uint64_t,3)
        REFER_DECL(uint64_t,4)
        REFER_DECL(float,1)
        REFER_DECL(float,2)
        REFER_DECL(float,3)
        REFER_DECL(float,4)
        REFER_DECL(double,1)
        REFER_DECL(double,2)
        REFER_DECL(double,3)
        REFER_DECL(double,4)
        REFER_DECL(std::complex<float>,1)
        REFER_DECL(std::complex<float>,2)
        REFER_DECL(std::complex<float>,3)
        REFER_DECL(std::complex<float>,4)
        REFER_DECL(std::complex<double>,1)
        REFER_DECL(std::complex<double>,2)
        REFER_DECL(std::complex<double>,3)
        REFER_DECL(std::complex<double>,4)


    template<int D> void Arrayset::appendCheck() const
    {
      if( D != m_n_dim ) {
        TDEBUG3("D=" << D << " -- Blitz array of size D=" << m_n_dim);
        error << "Cannot appened to the Arrayset a blitz array with a " <<
          "different number of dimensions." << std::endl;
        throw NDimensionError();
      }
    }


    template <typename T, int D>
    void Arrayset::append( const blitz::Array<T,D>& bl)
    {
      appendCheck<D>();

      boost::shared_ptr<Array> array(new Array(*this));
      // Find an available id and assign it to the Array
      // TODO: set id properly
      static size_t id = 157;//1;
      bool available_id = false;
      while( !available_id )
      {
        if(m_array.find(id) != m_array.end() )
          available_id = true;
        ++id;
      }
      array->setId(id);

      void* storage;
      array->setIsLoaded(true);

      // Check that the memory is contiguous in the blitz array
      // as this is required by the copy
      blitz::Array<T,D> ref;
      if( !bl.isStorageContiguous() )
        ref.reference(bl.copy());
      else
        ref.reference(bl);
      // Allocate storage area and copy the data from the blitz 
      // array to the storage area
      switch(m_element_type) {
        case array::t_bool:
          storage=new bool[m_n_elem];
          array->copyCast<bool,T>(ref.data()); break;
        case array::t_int8:
          storage=new int8_t[m_n_elem];
          array->copyCast<int8_t,T>(ref.data()); break;
        case array::t_int16:
          storage=new int16_t[m_n_elem];
          array->copyCast<int16_t,T>(ref.data()); break;
        case array::t_int32:
          storage=new int32_t[m_n_elem];
          array->copyCast<int32_t,T>(ref.data()); break;
        case array::t_int64:
          storage=new int64_t[m_n_elem];
          array->copyCast<int64_t,T>(ref.data()); break;
        case array::t_uint8:
          storage=new uint8_t[m_n_elem];
          array->copyCast<uint8_t,T>(ref.data()); break;
        case array::t_uint16:
          storage=new uint16_t[m_n_elem];
          array->copyCast<uint16_t,T>(ref.data()); break;
        case array::t_uint32:
          storage=new uint32_t[m_n_elem];
          array->copyCast<uint32_t,T>(ref.data()); break;
        case array::t_uint64:
          storage=new uint64_t[m_n_elem];
          array->copyCast<uint64_t,T>(ref.data()); break;
        case array::t_float32:
          storage=new float[m_n_elem];
          array->copyCast<float,T>(ref.data()); break;
        case array::t_float64:
          storage=new double[m_n_elem];
          array->copyCast<double,T>(ref.data()); break;
        case array::t_float128:
          storage=new long double[m_n_elem];
          array->copyCast<long double,T>(ref.data()); break;
        case array::t_complex64:
          storage=new std::complex<float>[m_n_elem];
          array->copyCast<std::complex<float>,T>(ref.data()); break;
        case array::t_complex128:
          storage=new std::complex<double>[m_n_elem];
          array->copyCast<std::complex<double>,T>(ref.data()); break;
        case array::t_complex256:
          storage=new std::complex<long double>[m_n_elem];
          array->copyCast<std::complex<long double>,T>(ref.data()); break;
        default:
          break;
      }
    }


    template<typename T, int D> void 
    Arrayset::at(size_t id, blitz::Array<T,D>& output) const {
      boost::shared_ptr<Array> x = (m_array.find(id))->second;
      x->copy(output);
    }




  }
  /**
   * @}
   */
}

#endif /* TORCH5SPRO_CORE_ARRAYSET_H */

