/**
 * @file src/cxx/database/database/XMLWriter.h
 * @author <a href="mailto:Laurent.El-Shafey@idiap.ch">Laurent El Shafey</a>
 *
 * @brief An XML Writer for a Dataset
 */

#ifndef TORCH5SPRO_XML_WRITER_H 
#define TORCH5SPRO_XML_WRITER_H

#include <string>
#include <libxml/encoding.h>
#include <libxml/xmlwriter.h>

namespace Torch {   
  /**
   * \ingroup libdatabase_api
   * @{
   *
   */
  namespace database {

    //Some promises
    class Dataset;

    /**
     * @brief The main class for the XML parser
     */
    class XMLWriter {
      public:
        /**
         * @brief Constructor
         */
        XMLWriter();

        /**
         * @brief Destructor
         */
        ~XMLWriter();

        /**
         * @brief Write a Dataset to an XML file.
         */
        void write(const char *filename, const Dataset& dataset,
          bool content_inline=false);

      private:
        /**
         * @brief Return an XML node containing an Arrayset
         */
        xmlNodePtr writeArrayset( xmlDocPtr doc, 
          boost::shared_ptr<const Arrayset> a, 
          bool content_inline, int precision=10, bool scientific=false);
        /**
         * @brief Return an XML node containing an Array
         */
        xmlNodePtr writeArray( xmlDocPtr doc, boost::shared_ptr<const Array> a,
          bool content_inline, int precision=10, bool scientific=false);
        /**
         * @brief Write the (casted) data in the given stringstrean
         */
        template <typename T> void writeData( std::stringstream& content,
          const T* data, size_t len) {
          content << data[0];
          for(size_t i=1; i<len; ++i)
            content << " " << data[i];
        }

        /**
         * @brief Return an XML node containing a Relationset
         */
        xmlNodePtr writeRelationset( xmlDocPtr doc, const Relationset& r,
          bool content_inline);
        /**
         * @brief Return an XML node containing a Rule
         */
        xmlNodePtr writeRule( xmlDocPtr doc, const Rule& r);
        /**
         * @brief Return an XML node containing a Relation
         */
        xmlNodePtr writeRelation( xmlDocPtr doc, const Relation& r);
        /**
         * @brief Return an XML node containing a Member
         */
        xmlNodePtr writeMember( xmlDocPtr doc, const Member& m);
    };

  }

  /**
   * @}
   */
}

#endif /* TORCH5SPRO_XML_WRITER_H */

