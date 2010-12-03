/**
 * @file src/core/core/DatasetXML.h
 * @author <a href="mailto:Laurent.El-Shafey@idiap.ch">Laurent El Shafey</a>
 * @author <a href="mailto:andre.anjos@idiap.ch">Andre Anjos</a>
 *
 * @brief A torch representation of a DatasetXML
 */

#ifndef TORCH5SPRO_CORE_DATASET_XML_H 
#define TORCH5SPRO_CORE_DATASET_XML_H

#include "core/Dataset2.h"
#include <blitz/array.h>
#include "core/logging.h"

#include <libxml/parser.h>
#include <libxml/tree.h>


namespace Torch {   
  /**
   * \ingroup libcore_api
   * @{
   *
   */
  namespace core {

    class s_file_loader {
      public:
        s_file_loader(const std::string& f, const std::string& l) {
          st_filename.assign(f);
          st_loader.assign(l);
        }

        // Attributes
        std::string st_filename;
        std::string st_loader;
    };

    /**
     * @brief The arrayset XML class for an XML dataset
     */
    class ArraysetXML: public Arrayset {
      public:
        /**
         * @brief Constructor to build an arrayset from a node 
         * of an XML file.
         */
        ArraysetXML(const xmlNodePtr& node);

        /**
         * @brief Return the id of the arrayset
         */
        size_t getId() { return m_id; }

/*      // Virtual template method are not valid in C++
        // One possibility is to define explicitly the at functions while 
        // keeping them virtual. 
        //   virtual void at(size_t id, blitz::Array<bool,1>& output) const
        // For this purpose, the C++ code might eventually be generated.

        virtual template<typename T, int D> 
        void at(size_t id, blitz::Array<T,D>& output) const {}
        //1. check if data is cached. if not load on-the-fly, with the 
        //   expected arrayset type
        //1.1 verify what is the actual data type inside the arrayset 
        //    (e.g., a complex with 2 dimensions) and use *that* type 
        //    to load the data
        //2. With that data loaded, cast over the output.
        //2.1 output type == expected data type: hand over memory
        //2.2 output type != expected data type: use blitz::cast()

        //if D != ndim: throw error;

        //switch(type-inside-xml-file) {
        //case bool:
        //  switch(ndim-inside-xml) {
        //  case 1:
        //    m_bool_1.copy(id, output);
        //    break;
        //  case 2:
        //}
        //
        //

        throw error;
      }
*/
    private:
      void setArraydata(const xmlNodePtr& node);
/*
      dictionary<id, blitz::Array<bool, 1> > m_bool_1;
      dictionary<id, blitz::Array<bool, 2> > m_bool_2;
      dictionary<id, blitz::Array<bool, 3> > m_bool_3;
      ...
      dictionary<id, blitz::Array<std::complex<long double>, 4> > m_complex256_4;
*/
      template<typename T1, typename T2, int D> void copy(blitz::Array<T1,D>& db_array, blitz::Array<T2,D>& user_array) {
//        user_array(blitz::cast(db_array));
      }
/*    // useless? make a copy to avoid modification of the data by the user
      template<typename T, int D> void copy(blitz::Array<T,D>& db_array, blitz::Array<T,D>& user_array) {
        user_array.get_memory(db_array);
      }
*/

      // Attributes
      // TODO: I'm not sure where to put these attributes 
      //   Dataset or DatasetXML?
      /**
       * @brief The id of the arrayset
       */
      size_t m_id;

      /**
       * @brief The role of the arrayset
       */
      std::string m_role;

      /**
       * @brief The type of the arrayset
       */
      std::string m_elementtype;

      /**
       * @brief The number of dimensions of the arrayset
       */
      size_t m_nb_dim;

      /**
       * @brief The shape of the arrayset. There are at most 4 dimensions.
       */
      size_t m_shape[4];

      /**
       * @brief The eventual file containing the data of the arrayset.
       */
      std::string m_filename;

      /**
       * @brief The eventual loader.
       */
      std::string m_loader;

      /**
       * @brief Value indicating the type of the data from this arrayset.
       *          0:  unknown/not yet set
       *          1:  bool  1D
       *          2:  bool  2D
       *          3:  bool  3D
       *          4:  bool  4D
       *          5:  int8  1D
       *          6:  int8  2D
       *          7:  int8  3D
       *          8:  int8  4D
       *          9:  int16 1D
       *          10: int16 2D
       *          11: int16 3D
       *          12: int16 4D
       *          13: int32 1D
       *          14: int32 2D
       *          15: int32 3D
       *          16: int32 4D
       *          17: int64 1D
       *          18: int64 2D
       *          19: int64 3D
       *          20: int64 4D
       *          21: uint8  1D
       *          22: uint8  2D
       *          23: uint8  3D
       *          24: uint8  4D
       *          25: uint16 1D
       *          26: uint16 2D
       *          27: uint16 3D
       *          28: uint16 4D
       *          29: uint32 1D
       *          30: uint32 2D
       *          31: uint32 3D
       *          32: uint32 4D
       *          33: uint64 1D
       *          34: uint64 2D
       *          35: uint64 3D
       *          36: uint64 4D
       *          37: float32 1D
       *          38: float32 2D
       *          39: float32 3D
       *          40: float32 4D
       *          41: float64 1D
       *          42: float64 2D
       *          43: float64 3D
       *          44: float64 4D
       *          45: float128 1D
       *          46: float128 2D
       *          47: float128 3D
       *          48: float128 4D
       *          49: complex64 1D
       *          50: complex64 2D
       *          51: complex64 3D
       *          52: complex64 4D
       *          53: complex128 1D
       *          54: complex128 2D
       *          55: complex128 3D
       *          56: complex128 4D
       *          57: complex256 1D
       *          58: complex256 2D
       *          59: complex256 3D
       *          60: complex256 4D
       */
      size_t m_blitz_type;

      /**
       * @brief The maps containing the Blitz++ arrays
       */
      std::map<size_t, blitz::Array<bool,1> > m_data_bool_1;
      std::map<size_t, blitz::Array<bool,2> > m_data_bool_2;
      std::map<size_t, blitz::Array<bool,3> > m_data_bool_3;
      std::map<size_t, blitz::Array<bool,4> > m_data_bool_4;
      std::map<size_t, blitz::Array<int8_t,1> > m_data_int8_1;
      std::map<size_t, blitz::Array<int8_t,2> > m_data_int8_2;
      std::map<size_t, blitz::Array<int8_t,3> > m_data_int8_3;
      std::map<size_t, blitz::Array<int8_t,4> > m_data_int8_4;
      std::map<size_t, blitz::Array<int16_t,1> > m_data_int16_1;
      std::map<size_t, blitz::Array<int16_t,2> > m_data_int16_2;
      std::map<size_t, blitz::Array<int16_t,3> > m_data_int16_3;
      std::map<size_t, blitz::Array<int16_t,4> > m_data_int16_4;
      std::map<size_t, blitz::Array<int32_t,1> > m_data_int32_1;
      std::map<size_t, blitz::Array<int32_t,2> > m_data_int32_2;
      std::map<size_t, blitz::Array<int32_t,3> > m_data_int32_3;
      std::map<size_t, blitz::Array<int32_t,4> > m_data_int32_4;
      std::map<size_t, blitz::Array<int64_t,1> > m_data_int64_1;
      std::map<size_t, blitz::Array<int64_t,2> > m_data_int64_2;
      std::map<size_t, blitz::Array<int64_t,3> > m_data_int64_3;
      std::map<size_t, blitz::Array<int64_t,4> > m_data_int64_4;
      std::map<size_t, blitz::Array<uint8_t,1> > m_data_uint8_1;
      std::map<size_t, blitz::Array<uint8_t,2> > m_data_uint8_2;
      std::map<size_t, blitz::Array<uint8_t,3> > m_data_uint8_3;
      std::map<size_t, blitz::Array<uint8_t,4> > m_data_uint8_4;
      std::map<size_t, blitz::Array<uint16_t,1> > m_data_uint16_1;
      std::map<size_t, blitz::Array<uint16_t,2> > m_data_uint16_2;
      std::map<size_t, blitz::Array<uint16_t,3> > m_data_uint16_3;
      std::map<size_t, blitz::Array<uint16_t,4> > m_data_uint16_4;
      std::map<size_t, blitz::Array<uint32_t,1> > m_data_uint32_1;
      std::map<size_t, blitz::Array<uint32_t,2> > m_data_uint32_2;
      std::map<size_t, blitz::Array<uint32_t,3> > m_data_uint32_3;
      std::map<size_t, blitz::Array<uint32_t,4> > m_data_uint32_4;
      std::map<size_t, blitz::Array<uint64_t,1> > m_data_uint64_1;
      std::map<size_t, blitz::Array<uint64_t,2> > m_data_uint64_2;
      std::map<size_t, blitz::Array<uint64_t,3> > m_data_uint64_3;
      std::map<size_t, blitz::Array<uint64_t,4> > m_data_uint64_4;
      std::map<size_t, blitz::Array<float,1> > m_data_float32_1;
      std::map<size_t, blitz::Array<float,2> > m_data_float32_2;
      std::map<size_t, blitz::Array<float,3> > m_data_float32_3;
      std::map<size_t, blitz::Array<float,4> > m_data_float32_4;
      std::map<size_t, blitz::Array<double,1> > m_data_float64_1;
      std::map<size_t, blitz::Array<double,2> > m_data_float64_2;
      std::map<size_t, blitz::Array<double,3> > m_data_float64_3;
      std::map<size_t, blitz::Array<double,4> > m_data_float64_4;
      std::map<size_t, blitz::Array<long double,1> > m_data_float128_1;
      std::map<size_t, blitz::Array<long double,2> > m_data_float128_2;
      std::map<size_t, blitz::Array<long double,3> > m_data_float128_3;
      std::map<size_t, blitz::Array<long double,4> > m_data_float128_4;
      std::map<size_t, blitz::Array<std::complex<float>,1> > 
        m_data_complex64_1;
      std::map<size_t, blitz::Array<std::complex<float>,2> > 
        m_data_complex64_2;
      std::map<size_t, blitz::Array<std::complex<float>,3> > 
        m_data_complex64_3;
      std::map<size_t, blitz::Array<std::complex<float>,4> > 
        m_data_complex64_4;
      std::map<size_t, blitz::Array<std::complex<double>,1> > 
        m_data_complex128_1;
      std::map<size_t, blitz::Array<std::complex<double>,2> > 
        m_data_complex128_2;
      std::map<size_t, blitz::Array<std::complex<double>,3> > 
        m_data_complex128_3;
      std::map<size_t, blitz::Array<std::complex<double>,4> > 
        m_data_complex128_4;
      std::map<size_t, blitz::Array<std::complex<long double>,1> > 
        m_data_complex256_1;
      std::map<size_t, blitz::Array<std::complex<long double>,2> > 
        m_data_complex256_2;
      std::map<size_t, blitz::Array<std::complex<long double>,3> > 
        m_data_complex256_3;
      std::map<size_t, blitz::Array<std::complex<long double>,4> > 
        m_data_complex256_4;


      /**
       * @brief The maps associating id with filenames/loader
       */
      std::map<size_t,s_file_loader> m_data_filenames;

    };

    /**
     * @brief The relation XML class for an XML dataset
     */
    class RelationXML: public Relation {
    };

    /**
     * @brief The rule XML class for an XML dataset
     */
    class RuleXML: public Rule {
    };

    /**
     * @brief The relationset XML class for an XML dataset
     */
    class RelationsetXML: public Relationset {
    };


    /**
     * @brief The main XML dataset class
     */
    class DatasetXML: public Dataset {
      public:
        /**
         * @brief Public constructor to build a dataset from an XML file.
         */
        DatasetXML(char *filename);

        /**
         * @brief Destructor
         */
        ~DatasetXML();

        /**
         * @brief The iterator class for a dataset
         */
        class const_iteratorXML;
        friend class const_iteratorXML;

        class const_iteratorXML: public const_iterator {
          public:
            const_iteratorXML():m_it(0) { }
            const_iteratorXML(const const_iteratorXML& it):
              m_it(it.m_it) { }
            ~const_iteratorXML() { }

            const_iteratorXML& operator=(const_iteratorXML& it) {
              m_it=it.m_it;
              return (*this);
            } 

            bool operator==(const const_iteratorXML& it) {
              return (m_it == it.m_it);
            } 

            bool operator!=(const const_iteratorXML& it) {
              return (m_it != it.m_it);
            } 

            const_iteratorXML& operator++() {
              m_it++;
              return (*this);
            } 

            const_iteratorXML& operator--() {
              m_it--;
              return (*this);
            }

            const ArraysetXML& operator*() {
              return *m_it->second;
            } 

          private:
            std::map<size_t, const ArraysetXML* >::const_iterator m_it;
        };
     
        /**
         *  @brief Iterators to access the arraysets contained in the dataset
         */
        virtual const_iterator begin() const;
        virtual const_iterator end() const;

        /**
         * @brief Return an arrayset with a given id
         * Throw an exception if there is no such arrayset
         */
        virtual const Arrayset& at( const size_t id ) const;

      private:
        /**
         * @brief Parse an XML file and update the dataset structure
         */
        void parseFile(char *filename);


        // Attributes
        /**
         * @brief Structure describing an XML document
         */
        xmlDocPtr m_doc;

        /**
         * @brief A container mapping ids to arraysets
         */
        std::map<size_t, ArraysetXML* > m_arrayset;
    };


  }
  /**
   * @}
   */
}

#endif /* TORCH5SPRO_CORE_DATASET_XML_H */

