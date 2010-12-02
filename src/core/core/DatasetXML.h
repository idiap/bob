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

        virtual template<typename T, int D> void at(size_t id, blitz::Array<T,D>& output) const {
        //1. check if data is cached. if not load on-the-fly, with the expected arrayset type
        //1.1 verify what is the actual data type inside the arrayset (e.g., a complex with 2 dimensions)
        //    and use *that* type to load the data
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
       * @brief The shape of the arrayset. There are at most 4 dimensions.
       */
      size_t m_shape[4];

      /**
       * @brief The eventual file containing the data of the arrayset.
       */
      std::string m_filename;
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
        std::map<size_t, const ArraysetXML* > m_arrayset;
    };


  }
  /**
   * @}
   */
}

#endif /* TORCH5SPRO_CORE_DATASET_XML_H */

