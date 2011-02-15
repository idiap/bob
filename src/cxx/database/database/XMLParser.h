/**
 * @file src/cxx/database/database/XMLParser.h
 * @author <a href="mailto:Laurent.El-Shafey@idiap.ch">Laurent El Shafey</a>
 * @author <a href="mailto:andre.anjos@idiap.ch">Andre Anjos</a>
 *
 * @brief An XML Parser for a Dataset
 */

#ifndef TORCH_DATABASE_XML_PARSER_H 
#define TORCH_DATABASE_XML_PARSER_H

#include <string>
#include <map>
#include <libxml/parser.h>
#include <libxml/tree.h>
#include <boost/tokenizer.hpp>
#include <blitz/array.h>

#include "core/Exception.h"
#include "core/logging.h"

namespace Torch {   
  /**
   * \ingroup libdatabase_api
   * @{
   *
   */
  namespace database {

    //Some promises
    class Dataset;
    class Arrayset;
    class Array;
    //class Relationset;
    //class Rule;
    //class Member;
    //class Relation;

    namespace detail {

      class XMLException: public Torch::core::Exception { };

      /**
       * @brief The main class for the XML parser
       */
      class XMLParser {
        public:
          /**
           * @brief Constructor
           */
          XMLParser();

          /**
           * @brief Destructor
           */
          ~XMLParser();


          /**
           * @brief Parse an XML file and update the dataset object accordingly
           * There are three check levels:
           *   + 0: libxml2 schema validation only
           *   + 1: same as 0 and high-level check
           *   + 2: same as 1 and high-level check while loading everything
           */
          void load(const char *filename, Dataset& dataset, 
              size_t check_level=1);


        private:
          /**
           * @brief Validation of the XML file against the XML Schema
           */
          void validateXMLSchema(xmlDocPtr doc);

          /**
           * @brief Parse an arrayset given an XML node and return the 
           * corresponding object.
           */
          boost::shared_ptr<Arrayset> parseArrayset( xmlNodePtr node);

          /**
           * @brief Parse a relationset given an XML node and return the 
           * corresponding object.
           */
//        boost::shared_ptr<Relationset> parseRelationset(xmlNodePtr node);

          /**
           * @brief Parse an array given an XML node and return the 
           * corresponding object.
           */
          boost::shared_ptr<Array> parseArray( xmlNodePtr node, 
            Torch::core::array::ElementType elem,
            size_t shape[Torch::core::array::N_MAX_DIMENSIONS_ARRAY], 
            size_t nb_dim );

          /**
           * @brief Parse a rule given an XML node and return the 
           * corresponding object.
           */
//        boost::shared_ptr<Rule> parseRule(xmlNodePtr node);

          /**
           * @brief Parse a relation given an XML node and return the 
           * corresponding object.
           */
//        boost::shared_ptr<Relation> parseRelation(xmlNodePtr node);

          /**
           * @brief Parse the data of an array given a tokenized string, and
           * check that the number of tokens matches the number of expected 
           * values.
           */
          template <typename T,int D> blitz::Array<T,D> parseArrayData(
            boost::tokenizer<boost::char_separator<char> > tok,
            const blitz::TinyVector<int,D> shape);
            /**
           * @brief Parse a member given an XML node and return the 
           * corresponding object.
           */
//        boost::shared_ptr<Member> parseMember(xmlNodePtr node);


          /**
           * @brief Mapping from arrayset-id to role
           */
//        boost::shared_ptr<std::map<size_t, std::string > > m_id_role;
      };


      /********************** TEMPLATE FUNCTION DEFINITIONS ***************/
      template <typename T,int D> blitz::Array<T,D> XMLParser::parseArrayData(
        boost::tokenizer<boost::char_separator<char> > tok,
        const blitz::TinyVector<int,D> shape)
//        size_t shape[Torch::core::array::N_MAX_DIMENSIONS_ARRAY], 
//        size_t nb_dim)
      {
        blitz::Array<T,D> array(shape);
        T* data_array = array.data();
        blitz::sizeType count = 0;
        for( boost::tokenizer<boost::char_separator<char> >::iterator
            it=tok.begin(); it!=tok.end(); ++it, ++count ) 
        {
          data_array[count] = boost::lexical_cast<T>(*it);
          TDEBUG3(data_array[count]);
        }

        if(count != array.numElements()) {
          Torch::core::error << "The number of values read (" << count <<
            ") in the array does not match with the expected number (" << 
            array.numElements() << ")" << std::endl;
          throw Torch::core::Exception();
        }

        return array;
      }


    }
  }

  /**
    * @}
    */
}

#endif /* TORCH_DATABASE_XML_PARSER_H */

