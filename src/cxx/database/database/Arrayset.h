/**
 * @file src/cxx/database/database/Arrayset.h
 * @author <a href="mailto:Laurent.El-Shafey@idiap.ch">Laurent El Shafey</a>
 *
 * @brief A torch representation of an Array for a Dataset.
 */

#ifndef TORCH5SPRO_DATABASE_ARRAYSET_H
#define TORCH5SPRO_DATABASE_ARRAYSET_H 1

#include <string>
#include <map>
#include <vector>
#include <boost/shared_ptr.hpp>
#include <boost/weak_ptr.hpp>
#include <blitz/array.h>

#include "core/logging.h"
#include "core/Exception.h"
#include "core/StaticComplexCast.h"
#include "database/dataset_common.h"

namespace Torch {   
  /**
   * \ingroup libdatabase_api
   * @{
   *
   */
  namespace database {
    
    class Dataset;
    /**
     * @brief The arrayset class for a dataset
     */
    class Arrayset {
      friend class XMLParser;
      friend class XMLWriter;
      friend class BinFile;

      public:
        /**
         * @brief Destructor
         */
        ~Arrayset();

        /*y*
         * @brief Add a copy of the given Array to the Arrayset
         */
        void append( boost::shared_ptr<const Array> array);
        /*y*
         * @brief Add a copy of the given Array to the Arrayset
         */
        void append( const Array& array);
        /**
         * @brief Add a blitz array to the Arrayset
         */
        template <typename T, int D>
        void append( const blitz::Array<T,D>& bl);
        /**
         * @brief Add a new array to the Arrayset
         */
        void append( const std::string& filename, const std::string& codec);
        /**
         * @brief Remove an Array with a given id from the Arrayset
         */
        void remove( const size_t id);

        /**
         * @brief Update the shape of the Array with the one given in the
         * blitz TinyVector.
         */
        template<int D> 
        void setShape( const blitz::TinyVector<int,D>& shape ) {
          m_n_dim = D;
          size_t n_elem = 1;
          for( int i=0; i<array::N_MAX_DIMENSIONS_ARRAY; ++i) {
            if( i< D) {
              m_shape[i] = shape(i);
              n_elem *= shape(i);
            } 
            else
              m_shape[i] = 0;
          }
          m_n_elem = n_elem;
        }

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
//        void setIsLoaded(const bool is_loaded) { m_is_loaded = is_loaded; }
        /**
         * @brief Set the filename containing the data if any. An empty string
         * indicates that the data are stored in the XML file directly.
         */
        void setFilename(const std::string& filename, const std::string& codec="")
          { m_filename.assign(filename); }
        /**
         * @brief Set the loader used to read the data from the external file 
         * if any.
         */
//        void setLoader(const LoaderType loader) { m_loader = loader; }
        
        /**
         * @brief Get the id of the Arrayset
         */
        size_t getId() const { return m_id; }
        /**
         * @brief Get the number of dimensions of the arrays of this Arrayset
         */
        size_t getNDim() const { return m_n_dim; }
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
        const std::string& getCodecname() const { return m_codecname; }
        /**
         * @brief Get the number of arrays in this Arrayset
         */
        size_t getNArrays() const { return m_array.size(); }

        /**
          * @brief Update the given vector with the 
          */
        void index( std::vector<size_t>& x );
        /**
         * @brief Return the array of the given index
         * @warning Please note that if you use that method, scope matters,
         * because the dataset owns the arrays.
         */
        const Array& operator[](size_t ind) const;
        Array& operator[](size_t ind);
        /**
         * @brief Return a smart pointer to the array of the given index
         */
        boost::shared_ptr<const Array> getArray(size_t ind) const;
        boost::shared_ptr<Array> getArray(size_t ind);

        /**
         * @brief Return a blitz array with the content of the array 
         * of the provided id.
         */
        template<typename T, int D> 
        blitz::Array<T,D> at(size_t id) const;


        boost::shared_ptr<const Dataset> getParentDataset() const;

        /** 
         * @brief const_iterator over the Arrays of the Dataset
         */
        typedef std::vector<boost::shared_ptr<Array> >::const_iterator
          const_iterator;
        /** 
         * @brief Return a const_iterator pointing at the first Array of 
         * the Dataset
         */
        const_iterator begin() const { return m_array.begin(); }
        /** 
         * @brief Return a const_iterator pointing at the last Array of 
         * the Dataset
         */
        const_iterator end() const { return m_array.end(); }

        /** 
         * @brief iterator over the Arrays of the Dataset
         */
        typedef std::vector<boost::shared_ptr<Array> >::iterator 
          iterator;
        /** 
         * @brief Return an iterator pointing at the first Array of 
         * the Dataset
         */
        iterator begin() { return m_array.begin(); }
        /** 
         * @brief Return an iterator pointing at the first Array of 
         * the Dataset
         */
        iterator end() { return m_array.end(); }



      private:
        /**
         * @brief Constructor
         */
        Arrayset( boost::shared_ptr<const Dataset> parent, const size_t id=0,
          const std::string& filename="", const std::string& codec="");

        /**
         * @brief Set the id of the Arrayset
         */
//        const size_t getId() { return m_id; }
        /**
         * @brief Set the number of dimensions of the arrays of this Arrayset
         */
//        int getNDim() { return m_n_dim; }
//      void setNDim(const size_t n_dim) { m_n_dim = n_dim; }
        /**
         * @brief Set the size of each dimension of the arrays of this 
         * Arrayset
         */
/*      void setShape(const size_t shape[]) { 
          for(size_t i=0; i<array::N_MAX_DIMENSIONS_ARRAY; ++i)
            m_shape[i] = shape[i];
        }*/

        /**
         * @brief Check that the blitz array has a compatible number of 
         * dimensions with this Arrayset
         */
        template <int D> void appendCheck() const;

        /**
          * @brief Return the shape in the C-style format
          */
        const size_t* getShape() const { return m_shape; }
        /**
         * @brief Update the shape of the Array with the one given.
         */
        void setShape( const size_t shape[array::N_MAX_DIMENSIONS_ARRAY] ) {
          m_n_dim = 0;
          size_t n_elem = 1;
          bool over = false;
          for( size_t i=0; i<array::N_MAX_DIMENSIONS_ARRAY; ++i) {
            if( !over && shape[i] > 0 ) {
              m_shape[i] = shape[i];
              n_elem *= shape[i];
              ++m_n_dim;
            } 
            else {
              over = true;
              m_shape[i] = 0;
            }
          }
          m_n_elem = n_elem;
        }


        /**
          * Attributes
          */
        boost::weak_ptr<const Dataset> m_parent_dataset;
        size_t m_id;

        size_t m_n_dim;
        size_t m_shape[array::N_MAX_DIMENSIONS_ARRAY];
        size_t m_n_elem;
        array::ElementType m_element_type;
        
        std::string m_role;
        bool m_is_loaded;
        std::string m_filename;
        std::string m_codecname;


        std::vector<boost::shared_ptr<Array> > m_array;
        std::map<size_t,size_t> m_index; //Key:id, value:index
    };



    /********************** TEMPLATE FUNCTION DEFINITIONS ***************/
    template <typename T, typename U> 
    void Array::copyCast( U* out ) const {
      size_t n_elem = (m_parent_arrayset.lock())->getNElem();
      for( size_t i=0; i<n_elem; ++i) {
        T* t_storage = reinterpret_cast<T*>(m_storage);
        static_complex_cast( t_storage[i], out[i] );
      }
    }

    template <typename T, int D> 
    blitz::Array<T,D> Array::data( ) const 
    {
      boost::shared_ptr<const Arrayset> parent = m_parent_arrayset.lock();
      if( D != parent->getNDim() ) {
        TDEBUG3("D=" << D << " -- ParseXML: D=" << 
          parent->getNDim());
        error << "Cannot copy the data in a blitz array with a different " <<
          "number of dimensions." << std::endl;
        throw NDimensionError();
      }

      // Reshape each dimensions with the correct size
      blitz::TinyVector<int,D> shape;
      parent->getShape(shape);

      blitz::Array<T,D> bl(shape);

      // Load the data if required
      if(!m_is_loaded) {
        ;//TODO:load 
        Array* a=const_cast<Array*>(this);
        a->m_is_loaded = true;
      }

      T* out_data = bl.data();
      switch(parent->getElementType()) {
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

      return bl;
    }

    template <int D> void Array::referCheck( ) const
    {
      // Load the data if required
      if(!m_is_loaded) {
        ;//TODO:load 
        Array* a=const_cast<Array*>(this);
        a->m_is_loaded = true;
      }

      boost::shared_ptr<const Arrayset> parent = m_parent_arrayset.lock();
      if( D != parent->getNDim() ) {
        TDEBUG3("D=" << D << " -- ParseXML: D=" <<
           parent->getNDim());
        error << "Cannot refer to the data in a blitz array with a " <<
          "different number of dimensions." << std::endl;
        throw NDimensionError();
      }
    }


#define REFER_DECL(T,D) template<> \
   blitz::Array<T,D> Array::data(); \

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

      // Find an available id and assign it to the Array
      // TODO: check that this works
      static size_t id = 1;
      bool available_id = false;
      while( !available_id )
      {
        if(m_index.find(id) == m_index.end() )
          available_id = true;
        ++id;
      }
      boost::shared_ptr<Array> array(
        new Array(boost::shared_ptr<const Arrayset>(this), id));

      void* storage;

      // Check that the memory is contiguous in the blitz array
      // as this is required by the copy
      blitz::Array<T,D> ref;
      if( !checkSafedata(bl) )
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

      // Update the m_is_loaded member of the array
      array->m_is_loaded = true;
    }


    template<typename T, int D> blitz::Array<T,D>
    Arrayset::at(size_t ind) const {
      boost::shared_ptr<const Array> x = m_array[ind];
      return x->data<T,D>();
    }




  }
  /**
   * @}
   */
}

#endif /* TORCH5SPRO_DATABASE_ARRAYSET_H */

