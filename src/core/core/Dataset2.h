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
  
    typedef enum Array_Type { t_unknown, t_bool, 
      t_int8, t_int16, t_int32, t_int64, 
      t_uint8, t_uint16, t_uint32, t_uint64, 
      t_float32, t_float64, t_float128,
      t_complex64, t_complex128, t_complex256 } Array_Type;

    typedef enum Loader_Type { l_unknown, l_blitz, l_tensor, l_bindata } 
      Loader_Type;

    
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
        Array(const boost::shared_ptr<Arrayset>& parent);
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
        void setIs_loaded(const bool is_loaded) { m_is_loaded = is_loaded; }
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
        void setLoader(const Loader_Type loader) { m_loader = loader; }
        /**
         * @brief Set the data of the Array. Storage should have been allocated
         * with malloc, to make the deallocation easy? 
         */
        void setStorage(void* storage) { m_storage = storage; }

        /**
         * @brief Get the id of the Array
         */
        size_t getId() const { return m_id; }
        /**
         * @brief Get the flag indicating if the array is loaded from an 
         * external file.
         */
        bool getIs_loaded() const { return m_is_loaded; }
        /**
         * @brief Get the filename containing the data if any. An empty string
         * indicates that the data is stored in the XML file directly.
         */
        const std::string& getFilename() const { return m_filename; }
        /**
         * @brief Get the loader used to read the data from the external file 
         * if any.
         */
        Loader_Type getLoader() const {return m_loader; }

        //TODO: method to get the data???

      private:
        const boost::shared_ptr<Arrayset>& m_parent_arrayset;
        size_t m_id;
        bool m_is_loaded;
        std::string m_filename;
        Loader_Type m_loader;
        void* m_storage;
        // The following member is duplicated from the parent arrayset. This
        // is necessary, as the parent arrayset is deleted before the 
        // underlying arrays, and type is required to cast the void pointer 
        // containing the data
        Array_Type m_element_type;
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
        void add_array( boost::shared_ptr<Array> array);

        /**
         * @brief Set the id of the Arrayset
         */
        void setId(const size_t id) { m_id = id; }
        /**
         * @brief Set the number of dimensions of the arrays of this Arrayset
         */
        void setN_dim(const size_t n_dim) { m_n_dim = n_dim; }
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
        void setN_elem(const size_t n_elem) {  m_n_elem = n_elem; } 
        /**
         * @brief Set the type of the elements contained in the the arrays of 
         * this Arrayset
         */
        void setArray_Type(const Array_Type element_type) 
          { m_element_type = element_type; }
        /**
         * @brief Set the role of the Arrayset
         */
        void setRole(const std::string& role) {m_role.assign(role); } 
        /**
         * @brief Set the flag indicating if this arrayset is loaded from an 
         * external file.
         */
        void setIs_loaded(const bool is_loaded) { m_is_loaded = is_loaded; }
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
        void setLoader(const Loader_Type loader) { m_loader = loader; }
        
        /**
         * @brief Get the id of the Arrayset
         */
        size_t getId() const { return m_id; }
        /**
         * @brief Get the number of dimensions of the arrays of this Arrayset
         */
        size_t getN_dim() const { return m_n_dim; }
        /**
         * @brief Get the size of each dimension of the arrays of this 
         * Arrayset
         */
        const size_t* getShape() const { return m_shape; }
        /**
         * @brief Get the number of elements in each array of this 
         * Arrayset
         */
        const size_t getN_elem() const { return m_n_elem; } 
        /**
         * @brief Get the type of the elements contained in the the arrays of 
         * this Arrayset
         */
        Array_Type getArray_Type() const { return m_element_type; }
        /**
         * @brief Get the role of this Arrayset
         */
        const std::string& getRole() const { return m_role; }
        /**
         * @brief Get the flag indicating if the arrayset is loaded from an 
         * external file.
         */
        bool getIs_loaded() const { return m_is_loaded; }
        /**
         * @brief Get the filename containing the data if any. An empty string
         * indicates that the data is stored in the XML file directly.
         */
        const std::string& getFilename() const { return m_filename; }
        /**
         * @brief Get the loader used to read the data from the external file 
         * if any.
         */
        Loader_Type getLoader() const {return m_loader; }


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
        template<typename T, int D> void
           at(size_t id, blitz::Array<T,D>& output);
        // blitz::Array<float, 2> myarray;
        // arrayset->at(3, myarray);
        //
      private:
        size_t m_id;

        size_t m_n_dim;
        size_t m_shape[4];
        size_t m_n_elem;
        Array_Type m_element_type;
        
        std::string m_role;
        bool m_is_loaded;
        std::string m_filename;
        Loader_Type m_loader;

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
        void add_arrayset( boost::shared_ptr<Arrayset> arrayset);

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


  }
  /**
   * @}
   */
}

#endif /* TORCH5SPRO_CORE_DATASET_H */

