/**
 * @file src/core/src/XMLParser.cc
 * @author <a href="mailto:Laurent.El-Shafey@idiap.ch">Laurent El Shafey</a>
 *
 * @brief Implements the XML parser for a dataset.
 */

#include "core/XMLParser.h"
#include "core/Exception.h"

#include <libxml/xmlschemas.h>

namespace Torch {
  namespace core {

    /**
     * string for the XML attributes
     */
    namespace db {
      static const char dataset[]           = "dataset";
      static const char arrayset[]          = "arrayset";
      static const char external_arrayset[] = "external-arrayset";
      static const char id[]                = "id";
      static const char role[]              = "role";
      static const char elementtype[]       = "elementtype";
      static const char shape[]             = "shape";
      static const char loader[]            = "loader";
      static const char file[]              = "file";
      static const char array[]             = "array";
      static const char external_array[]    = "external-array";

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
    }



    XMLParser::XMLParser() { }


    XMLParser::~XMLParser() { }


    void XMLParser::validateXMLSchema(xmlDocPtr doc) {
      // Get path to the XML Schema definition
      char *schema_path = getenv("TORCH_SCHEMA_PATH");
      if( !strcmp( schema_path, "") )
        warn << "Environment variable $TORCH_SCHEMA_PATH is not set." <<
          std::endl;
      char schema_full_path[1024];
      strcpy( schema_full_path, schema_path);
      strcat( schema_full_path, "/dataset.xsd" );

      // Load the XML schema from the file
      xmlDocPtr xsd_doc = xmlReadFile(schema_full_path, 0, 0);
      if(xsd_doc == 0) {
        error << "Unable to load the XML schema" << std::endl;
        throw Exception();        
      }
      // Create an XML schema parse context
      xmlSchemaParserCtxtPtr xsd_parser = xmlSchemaNewDocParserCtxt(xsd_doc);
      if(xsd_parser == 0) {
        xmlFreeDoc(xsd_doc);
        error << "Unable to create the XML schema parse context." << std::endl;
        throw Exception();        
      }
      // Parse the XML schema definition and check its correctness
      xmlSchemaPtr xsd_schema = xmlSchemaParse(xsd_parser);
      if(xsd_schema == 0) {
        xmlSchemaFreeParserCtxt(xsd_parser);
        xmlFreeDoc(xsd_doc); 
        error << "Invalid XML Schema definition." << std::endl;
        throw Exception();        
      }
      // Create an XML schema validation context for the schema
      xmlSchemaValidCtxtPtr xsd_valid = xmlSchemaNewValidCtxt(xsd_schema);
      if(xsd_valid == 0) {
        xmlSchemaFree(xsd_schema);
        xmlSchemaFreeParserCtxt(xsd_parser);
        xmlFreeDoc(xsd_doc);
        error << "Unable to create an XML Schema validation context." <<
          std::endl;
        throw Exception();
      }

      // Check that the document is valid wrt. to the schema, and throw an 
      // exception otherwise.
      if(xmlSchemaValidateDoc(xsd_valid, doc)) {
        error << "The XML file is NOT valid with respect to the XML schema." <<
          std::endl;
        throw Exception();
      }

      xmlSchemaFreeValidCtxt(xsd_valid);
      xmlSchemaFree(xsd_schema);
      xmlSchemaFreeParserCtxt(xsd_parser);
      xmlFreeDoc(xsd_doc);
    }


    void XMLParser::load(const char* filename, Dataset& dataset) {
      // Parse the XML file with libxml2
      xmlDocPtr doc = xmlParseFile(filename);
      xmlNodePtr cur; 

      // Check validity of the XML file
      if(doc == 0 ) {
        error << "Document " << filename << " was not parsed successfully." <<
          std::endl;
        throw Exception();
      }

      // Check that the XML file is not empty
      cur = xmlDocGetRootElement(doc);
      if(cur == 0) { 
        error << "Document " << filename << " is empty." << std::endl;
        xmlFreeDoc(doc);
        throw Exception();
      }

      // Check that the XML file contains a dataset
      if( strcmp((const char*)cur->name, db::dataset) ) {
        error << "Document " << filename << 
          " is of the wrong type (!= dataset)." << std::endl;
        xmlFreeDoc(doc);
        throw Exception();
      } 

      // Validate the XML document against the XML Schema
      // Throw an exception in case of failure
      validateXMLSchema( doc);


      // Parse Arraysets
      cur = cur->xmlChildrenNode;
      while(cur != 0) { 
        // Parse an arrayset and add it to the dataset
        if( !strcmp((const char*)cur->name, db::arrayset) || 
            !strcmp((const char*)cur->name, db::external_arrayset) )
          dataset.addArrayset( parseArrayset(cur) );
        cur = cur->next;
      }

      xmlFreeDoc(doc);
    }


    boost::shared_ptr<Arrayset> XMLParser::parseArrayset(const xmlNodePtr cur) {
      boost::shared_ptr<Arrayset> arrayset(new Arrayset());
      // Parse id
      xmlChar *str;
      str = xmlGetProp(cur, (const xmlChar*)db::id);
      arrayset->setId(str!=0? static_cast<size_t>(atoi((const char*)str)): 0);
      std::cout << "Id: " << arrayset->getId() << std::endl;
      xmlFree(str);

      // Parse role
      str = xmlGetProp(cur, (const xmlChar*)db::role);
      arrayset->setRole( ( (str!=0?(const char *)str:"") ) );
      std::cout << "Role: " << arrayset->getRole() << std::endl;
      xmlFree(str);

      // Parse elementtype
      str = xmlGetProp(cur, (const xmlChar*)db::elementtype);
      if( str==0 ) {
        error << "Elementtype is not specified in arrayset (id: " << 
          arrayset->getId() << ")." << std::endl;
        throw Exception();
      }
      std::string str_element_type( (const char*)str );
      if( !str_element_type.compare( db::t_bool ) )
        arrayset->setArrayType( t_bool );
      else if( !str_element_type.compare( db::t_uint8 ) )
        arrayset->setArrayType( t_uint8 );
      else if( !str_element_type.compare( db::t_uint16 ) )
        arrayset->setArrayType( t_uint16 );
      else if( !str_element_type.compare( db::t_uint32 ) )
        arrayset->setArrayType( t_uint32 );
      else if( !str_element_type.compare( db::t_uint64 ) )
        arrayset->setArrayType( t_uint64 );
      else if( !str_element_type.compare( db::t_int8 ) )
        arrayset->setArrayType( t_int8 );
      else if( !str_element_type.compare( db::t_int16 ) )
        arrayset->setArrayType( t_int16 );
      else if( !str_element_type.compare( db::t_int32 ) )
        arrayset->setArrayType( t_int32 );
      else if( !str_element_type.compare( db::t_int64 ) )
        arrayset->setArrayType( t_int64 );
      else if( !str_element_type.compare( db::t_float32 ) )
        arrayset->setArrayType( t_float32 );
      else if( !str_element_type.compare( db::t_float64 ) )
        arrayset->setArrayType( t_float64 );
      else if( !str_element_type.compare( db::t_float128 ) )
        arrayset->setArrayType( t_float128 );
      else if( !str_element_type.compare( db::t_complex64 ) )
        arrayset->setArrayType( t_complex64 );
      else if( !str_element_type.compare( db::t_complex128 ) )
        arrayset->setArrayType( t_complex128 );
      else if( !str_element_type.compare( db::t_complex256 ) )
        arrayset->setArrayType( t_complex256 );
      else
        arrayset->setArrayType( t_unknown );
      std::cout << "Elementtype: " << arrayset->getArrayType() << std::endl;
      xmlFree(str);

      // Parse shape
      size_t shape[4];
      shape[0]=shape[1]=shape[2]=shape[3]=0;
      str = xmlGetProp(cur, (const xmlChar*)db::shape);
      if( str==0 ) {
        error << "Elementtype is not specified in arrayset (id: " << 
          arrayset->getId() << ")." << std::endl;
        throw Exception();
      }
      // Tokenize the shape string to extract the dimensions
      std::string str_shape((const char *)str);
      boost::tokenizer<> tok(str_shape);
      size_t count=0;
      for( boost::tokenizer<>::iterator it=tok.begin(); it!=tok.end(); 
        ++it, ++count ) 
      {
        if(count>3) {
          error << "Shape is not valid in arrayset (id: " << 
            arrayset->getId() << "). Maximum number of dimensions is 4." << 
            std::endl;
          throw Exception();        
        }
        shape[count] = atoi((*it).c_str());
      }
      arrayset->setNDim(count);
      arrayset->setShape(shape);
      std::cout << "Nb dimensions: " << arrayset->getNDim() << std::endl;
      std::cout << "Shape: (" << arrayset->getShape()[0] << "," << 
        arrayset->getShape()[1] << ","<< arrayset->getShape()[2] << "," << 
        arrayset->getShape()[3] << ")" << std::endl;
      xmlFree(str);
      // Set the number of elements
      size_t n_elem = arrayset->getShape()[0];
      for( size_t i=1; i < arrayset->getNDim(); ++i)
        n_elem *= arrayset->getShape()[i];
      arrayset->setNElem(n_elem);

      // Parse loader
      str = xmlGetProp(cur, (const xmlChar*)db::loader);
      std::string str_loader( str!=0 ? (const char*)str: "" );
      if( !str_loader.compare( db::l_blitz ) )
        arrayset->setLoader( l_blitz );
      else if( !str_loader.compare( db::l_tensor ) )
        arrayset->setLoader( l_tensor );
      else if( !str_loader.compare( db::l_bindata ) )
        arrayset->setLoader( l_bindata );
      else 
        arrayset->setLoader( l_unknown );
      std::cout << "Loader: " << arrayset->getLoader() << std::endl;
      xmlFree(str);

      // Parse filename
      str = xmlGetProp(cur, (const xmlChar*)db::file);
      arrayset->setFilename( (str!=0?(const char*)str:"") );
      std::cout << "File: " << arrayset->getFilename() << std::endl;
      xmlFree(str);

      if( !arrayset->getFilename().compare("") )
      {
        // Parse the data
        xmlNodePtr cur_data = cur->xmlChildrenNode;

        while (cur_data != 0) { 
          // Process an array
          if ( !strcmp( (const char*)cur_data->name, db::array) || 
               !strcmp( (const char*)cur_data->name, db::external_array) ) {
            arrayset->addArray( parseArray( *arrayset, cur_data) );
          }
          cur_data = cur_data->next;
        }
      }

      return arrayset;
    }


    boost::shared_ptr<Array> XMLParser::parseArray(
      const Arrayset& parent, const xmlNodePtr cur) 
    {
      boost::shared_ptr<Array> array(new Array(parent));
      // Parse id
      xmlChar *str;
      str = xmlGetProp(cur, (const xmlChar*)db::id);
      array->setId(str!=0? static_cast<size_t>(atoi((const char*)str)): 0);
      std::cout << "  Array Id: " << array->getId() << std::endl;
      xmlFree(str);

      // Parse loader
      str = xmlGetProp(cur, (const xmlChar*)db::loader);
      std::string str_loader( str!=0 ? (const char*)str: "" );
      if( !str_loader.compare( db::l_blitz ) )
        array->setLoader( l_blitz );
      else if( !str_loader.compare( db::l_tensor ) )
        array->setLoader( l_tensor );
      else if( !str_loader.compare( db::l_bindata ) )
        array->setLoader( l_bindata );
      else 
        array->setLoader( l_unknown );
      std::cout << "  Array Loader: " << array->getLoader() << std::endl;
      xmlFree(str);

      // Parse filename
      str = xmlGetProp(cur, (const xmlChar*)db::file);
      array->setFilename( (str!=0?(const char*)str:"") );
      std::cout << "  Array File: " << array->getFilename() << std::endl;
      xmlFree(str);

      // Parse the data contained in the XML file
      if( !array->getFilename().compare("") )
      {
        // Preliminary for the processing of the content of the array
        xmlChar* content = xmlNodeGetContent(cur);
        std::string data( (const char *)content);
        boost::char_separator<char> sep(" ;|");
        boost::tokenizer<boost::char_separator<char> > tok(data, sep);
        xmlFree(content);

        // Switch over the possible type
        size_t nb_values = parent.getNElem();
        switch( parent.getArrayType()) {
          case t_bool:
            array->setStorage( parseArrayData<bool>( tok, nb_values ) );
            break;
          case t_int8:
            array->setStorage( parseArrayData<int8_t>( tok, nb_values ) );
            break;
          case t_int16:
            array->setStorage( parseArrayData<int16_t>( tok, nb_values ) );
            break;
          case t_int32:
            array->setStorage( parseArrayData<int32_t>( tok, nb_values ) );
            break;
          case t_int64:
            array->setStorage( parseArrayData<int64_t>( tok, nb_values ) );
            break;
          case t_uint8:
            array->setStorage( parseArrayData<uint8_t>( tok, nb_values ) );
            break;
          case t_uint16:
            array->setStorage( parseArrayData<uint16_t>( tok, nb_values ) );
            break;
          case t_uint32:
            array->setStorage( parseArrayData<uint32_t>( tok, nb_values ) );
            break;
          case t_uint64:
            array->setStorage( parseArrayData<uint64_t>( tok, nb_values ) );
            break;
          case t_float32:
            array->setStorage( parseArrayData<float>( tok, nb_values ) );
            break;
          case t_float64:
            array->setStorage( parseArrayData<double>( tok, nb_values ) );
            break;
          case t_float128:
            array->setStorage( parseArrayData<long double>( tok, nb_values ) );
            break;
          case t_complex64:
            array->setStorage( parseArrayData<std::complex<float> >( tok, 
              nb_values ) );
            break;
          case t_complex128:
            array->setStorage( parseArrayData<std::complex<double> >( tok, 
              nb_values ) );
            break;
          case t_complex256:
            array->setStorage( parseArrayData<std::complex<long double> >( 
              tok, nb_values ) );
            break;
          default:
            break;
        }
      }
      
      return array;
    }


  }
}

