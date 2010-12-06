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


        // TODO: Should we make this method static?
        /**
         * @brief Parse an XML file an return a Dataset. The returned dataset
         * should be deallocated by the caller using delete.
         */
        Dataset* load(const char *filename);


      private:
        // TODO: Should we make the following methods static?
        /**
         * @brief Parse an arrayset given an XML node and return the 
         * corresponding object.
         */
        boost::shared_ptr<Arrayset> parseArrayset(xmlNodePtr node);

        /**
         * @brief Parse an array given an XML node and return the 
         * corresponding object.
         */
        boost::shared_ptr<Array> parseArray(xmlNodePtr node, 
          Array_Type a_type, size_t nb_values);

        /**
         * @brief Parse the data of an array given a tokenized string, and
         * check that the number of tokens matches the number of expected 
         * values.
         */
        template <typename T> T* parseArrayData( 
          boost::tokenizer<boost::char_separator<char> > tok, 
          size_t nb_values );


        // Attributes
        /** 
         * @brief Structure describing an XML document
         */
        xmlDocPtr m_doc; // TODO: Necessary?
    };

  }

  /**
   * @}
   */
}

#endif /* TORCH5SPRO_XML_PARSER_H */

