/**
 * @file cxx/src/core/core/XMLParser.h
 * @author <a href="mailto:Laurent.El-Shafey@idiap.ch">Laurent El Shafey</a>
 * @author <a href="mailto:andre.anjos@idiap.ch">Andre Anjos</a>
 *
 * @brief An XML Parser for a Dataset
 */

#ifndef TORCH5SPRO_XML_PARSER_H 
#define TORCH5SPRO_XML_PARSER_H

#include <blitz/array.h>
#include "core/logging.h"

#include <libxml/parser.h>
#include <libxml/tree.h>

#include <string>

#include "core/Dataset2.h"


namespace Torch {   
  /**
   * \ingroup libcore_api
   * @{
   *
   */
  namespace core {

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
         * @brief Parse an XML file and update the dataset object accordingly.
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
        boost::shared_ptr<Arrayset> parseArrayset(xmlNodePtr node);

        /**
         * @brief Parse a relationset given an XML node and return the 
         * corresponding object.
         */
        boost::shared_ptr<Relationset> parseRelationset(xmlNodePtr node);

        /**
         * @brief Parse an array given an XML node and return the 
         * corresponding object.
         */
        boost::shared_ptr<Array> parseArray( 
          const Arrayset& parent, xmlNodePtr node);

        /**
         * @brief Parse a rule given an XML node and return the 
         * corresponding object.
         */
        boost::shared_ptr<Rule> parseRule(xmlNodePtr node);

        /**
         * @brief Parse a relation given an XML node and return the 
         * corresponding object.
         */
        boost::shared_ptr<Relation> parseRelation(xmlNodePtr node);

        /**
         * @brief Parse the data of an array given a tokenized string, and
         * check that the number of tokens matches the number of expected 
         * values.
         */
        template <typename T> T* parseArrayData( 
          boost::tokenizer<boost::char_separator<char> > tok, 
          size_t nb_values );

        /**
         * @brief Parse a member given an XML node and return the 
         * corresponding object.
         */
        boost::shared_ptr<Member> parseMember(xmlNodePtr node);


        /**
         * @brief Mapping from arrayset-id to role
         */
        boost::shared_ptr<std::map<size_t, std::string > > m_id_role;
    };


    /********************** TEMPLATE FUNCTION DEFINITIONS ***************/
    template <typename T> T* XMLParser::parseArrayData( 
      boost::tokenizer<boost::char_separator<char> > tok, size_t nb_values )
    {
      T* data_array = new T[nb_values];
      size_t count = 0;
      for( boost::tokenizer<boost::char_separator<char> >::iterator
          it=tok.begin(); it!=tok.end(); ++it, ++count ) 
      {
        data_array[count] = boost::lexical_cast<T>(*it);
        TDEBUG3(data_array[count]);
      }

      if(count != nb_values) {
        Torch::core::error << "The number of values read (" << count <<
          ") in the array does not match with the expected number (" << 
          nb_values << ")" << std::endl;
        throw Exception();
      }

      return data_array;
    }

    
  }

  /**
   * @}
   */
}

#endif /* TORCH5SPRO_XML_PARSER_H */

