/**
 * @file src/core/core/XMLParser.h
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
         */
        void load(const char *filename, Dataset& dataset);


      private:
        /**
         * @brief Parse an arrayset given an XML node and return the 
         * corresponding object.
         */
        boost::shared_ptr<Arrayset> parseArrayset(xmlNodePtr node);

        /**
         * @brief Parse an array given an XML node and return the 
         * corresponding object.
         */
        boost::shared_ptr<Array> parseArray( 
          const boost::shared_ptr<Arrayset> parent, xmlNodePtr node, 
          Array_Type a_type, size_t nb_values);

        /**
         * @brief Parse the data of an array given a tokenized string, and
         * check that the number of tokens matches the number of expected 
         * values.
         */
        template <typename T> T* parseArrayData( 
          boost::tokenizer<boost::char_separator<char> > tok, 
          size_t nb_values );

    };

  }

  /**
   * @}
   */
}

#endif /* TORCH5SPRO_XML_PARSER_H */

