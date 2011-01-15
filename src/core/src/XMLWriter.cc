/**
 * @file src/core/src/XMLWriter.cc
 * @author <a href="mailto:Laurent.El-Shafey@idiap.ch">Laurent El Shafey</a>
 *
 * @brief Implements the XML writer for a dataset.
 */

#include "core/XMLWriter.h"
#include "core/Exception.h"

namespace Torch {
  namespace core {

    // TODO: Remove duplication of the following strings
    /**
     * string for the XML attributes
     */
    namespace db {
      static const char dataset[]           = "dataset";
      static const char arrayset[]          = "arrayset";
      static const char external_arrayset[] = "external-arrayset";
      static const char relationset[]       = "relationset";
      static const char id[]                = "id";
      static const char role[]              = "role";
      static const char elementtype[]       = "elementtype";
      static const char shape[]             = "shape";
      static const char loader[]            = "loader";
      static const char file[]              = "file";
      static const char array[]             = "array";
      static const char external_array[]    = "external-array";
      static const char name[]              = "name";
      static const char rule[]              = "rule";
      static const char relation[]          = "relation";
      static const char member[]            = "member";
      static const char arrayset_member[]   = "arrayset-member";
      static const char arrayset_role[]     = "arrayset-role";
      static const char min[]               = "min";
      static const char max[]               = "max";
      static const char array_id[]          = "array-id";
      static const char arrayset_id[]       = "arrayset-id";

      // elementtype
      static const char t_bool[]        = "bool";
      static const char t_int8[]        = "int8";
      static const char t_int16[]       = "int16";
      static const char t_int32[]       = "int32";
      static const char t_int64[]       = "int64";
      static const char t_uint8[]       = "uint8";
      static const char t_uint16[]      = "uint16";
      static const char t_uint32[]      = "uint32";
      static const char t_uint64[]      = "uint64";
      static const char t_float32[]     = "float32";
      static const char t_float64[]     = "float64";
      static const char t_float128[]    = "float128";
      static const char t_complex64[]   = "complex64";
      static const char t_complex128[]  = "complex128";
      static const char t_complex256[]  = "complex256";

      // loader
      static const char l_blitz[]       = "blitz";
      static const char l_tensor[]      = "tensor";
      static const char l_bindata[]     = "bindata";
      static const char l_byextension[] = "byextension";
    }


    XMLWriter::XMLWriter() { }

    XMLWriter::~XMLWriter() { }

    void XMLWriter::write(const char *filename, const Dataset& dataset,
      bool b) 
    {
      xmlDocPtr doc;
      xmlNodePtr rootnode;

      doc = xmlNewDoc((const xmlChar*)"1.0");
      // Create the root node (Dataset) and set it in the document
      rootnode = xmlNewDocNode(doc, 0, (const xmlChar*)db::dataset, 0);
      xmlDocSetRootElement(doc, rootnode);

      // Create Arrayset nodes
      for(Dataset::const_iterator it=dataset.begin(); it!=dataset.end(); 
        ++it)
      {
        xmlAddChild( rootnode, writeArrayset( doc, *it->second, b) );
      }
      // Create Relationset nodes
      for(Dataset::relationset_const_iterator it=dataset.relationset_begin(); 
        it!=dataset.relationset_end(); ++it)
      {
        xmlAddChild( rootnode, writeRelationset( doc, *it->second, b) );
      }

      // Save the document to the specified XML file in UTF-8 format
      xmlSaveFormatFileEnc(filename, doc, "UTF-8", 1);
    }


    xmlNodePtr XMLWriter::writeArrayset( xmlDocPtr doc, const Arrayset& a, 
      bool content_inline) 
    {
      // TODO: implementation 
      return xmlNewDocNode(doc, 0, (const xmlChar*)db::arrayset, 0);
    }


    xmlNodePtr XMLWriter::writeRelationset( xmlDocPtr doc, const Relationset& r,
      bool content_inline) 
    {
      // TODO: implementation
      return xmlNewDocNode(doc, 0, (const xmlChar*)db::relationset, 0);
    }

  }
}

