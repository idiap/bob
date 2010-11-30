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
#include <libxml/parser.h>
#include <libxml/tree.h>
#include <blitz/array.h>


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
/*
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
/* USELESS: make a copy to avoid modification of the data by the user
      template<typename T, int D> void copy(blitz::Array<T,D>& db_array, blitz::Array<T,D>& user_array) {
        user_array.get_memory(db_array);
      }
*/

      // Attributes

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
        DatasetXML();
        DatasetXML(char *filename);
        ~DatasetXML();
        //query/iterate over:
        //1. "Array"
        //2. "ArraySet"
        //3. "TargetSet"
/*      
        virtual const_iterator begin() const;
        virtual const_iterator end() const;

        virtual const ArraySet& at (size_t id) const;
*/
      private:
        xmlDocPtr m_doc;
        
    };


  }
  /**
   * @}
   */
}

#endif /* TORCH5SPRO_CORE_DATASET_XML_H */

