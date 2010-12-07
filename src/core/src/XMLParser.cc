/**
 * @file src/core/src/XMLParser.cc
 * @author <a href="mailto:Laurent.El-Shafey@idiap.ch">Laurent El Shafey</a>
 *
 * @brief Implements the XML parser for a dataset.
 */

#include "core/XMLParser.h"
#include "core/Exception.h"

#include <sys/stat.h> 

namespace Torch {
  namespace core {

    /**
     * string for the XML attributes
     */
    namespace db {
      static const char dataset[]     = "dataset";
      static const char arrayset[]    = "arrayset";
      static const char id[]          = "id";
      static const char role[]        = "role";
      static const char elementtype[] = "elementtype";
      static const char shape[]       = "shape";
      static const char loader[]      = "loader";
      static const char file[]        = "file";
      static const char array[]       = "array";

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


    void XMLParser::load(const char* filename, Dataset& dataset) {
      // Parse the XML file with libxml2
      xmlDocPtr m_doc = xmlParseFile(filename);
      xmlNodePtr cur; 

      // Check validity of the XML file
      if(m_doc == 0 ) {
        error << "Document " << filename << " was not parsed successfully." <<
          std::endl;
        throw Exception();
      }

      // Check that the XML file is not empty
      cur = xmlDocGetRootElement(m_doc);
      if (cur == 0) { 
        error << "Document " << filename << " is empty." << std::endl;
        xmlFreeDoc(m_doc);
        throw Exception();
      }

      // Check that the XML file contains a dataset
      if (xmlStrcmp(cur->name, xmlCharStrdup(db::dataset))) {
        error << "Document " << filename << 
          " is of the wrong type (!= dataset)." << std::endl;
        xmlFreeDoc(m_doc);
        throw Exception();
      }    

      // Parse Arraysets
      cur = cur->xmlChildrenNode;
      while (cur != 0) { 
        // Parse an arrayset and add it to the dataset
        if ((!xmlStrcmp(cur->name, xmlCharStrdup(db::arrayset)))) 
          dataset.add_arrayset( parseArrayset(cur) );
        cur = cur->next;
      }

    }


    boost::shared_ptr<Arrayset> XMLParser::parseArrayset(const xmlNodePtr cur) {
      boost::shared_ptr<Arrayset> arrayset(new Arrayset());
      // Parse id
      xmlChar *str;
      str = xmlGetProp(cur, xmlCharStrdup(db::id));
      arrayset->setId(str!=0? static_cast<size_t>(atoi((const char*)str)): 0);
      std::cout << "Id: " << arrayset->getId() << std::endl;
      xmlFree(str);

      // Parse role
      str = xmlGetProp(cur, xmlCharStrdup(db::role));
      arrayset->setRole( ( (str!=0?(const char *)str:"") ) );
      std::cout << "Role: " << arrayset->getRole() << std::endl;
      xmlFree(str);

      // Parse elementtype
      str = xmlGetProp(cur, xmlCharStrdup(db::elementtype));
      if( str==0 ) {
        error << "Elementtype is not specified in arrayset (id: " << 
          arrayset->getId() << ")." << std::endl;
        throw Exception();
      }
      std::string str_element_type( (const char*)str );
      if( !str_element_type.compare( db::t_bool ) )
        arrayset->setArray_Type( t_bool );
      else if( !str_element_type.compare( db::t_uint8 ) )
        arrayset->setArray_Type( t_uint8 );
      else if( !str_element_type.compare( db::t_uint16 ) )
        arrayset->setArray_Type( t_uint16 );
      else if( !str_element_type.compare( db::t_uint32 ) )
        arrayset->setArray_Type( t_uint32 );
      else if( !str_element_type.compare( db::t_uint64 ) )
        arrayset->setArray_Type( t_uint64 );
      else if( !str_element_type.compare( db::t_int8 ) )
        arrayset->setArray_Type( t_int8 );
      else if( !str_element_type.compare( db::t_int16 ) )
        arrayset->setArray_Type( t_int16 );
      else if( !str_element_type.compare( db::t_int32 ) )
        arrayset->setArray_Type( t_int32 );
      else if( !str_element_type.compare( db::t_int64 ) )
        arrayset->setArray_Type( t_int64 );
      else if( !str_element_type.compare( db::t_float32 ) )
        arrayset->setArray_Type( t_float32 );
      else if( !str_element_type.compare( db::t_float64 ) )
        arrayset->setArray_Type( t_float64 );
      else if( !str_element_type.compare( db::t_float128 ) )
        arrayset->setArray_Type( t_float128 );
      else if( !str_element_type.compare( db::t_complex64 ) )
        arrayset->setArray_Type( t_complex64 );
      else if( !str_element_type.compare( db::t_complex128 ) )
        arrayset->setArray_Type( t_complex128 );
      else if( !str_element_type.compare( db::t_complex256 ) )
        arrayset->setArray_Type( t_complex256 );
      else
        arrayset->setArray_Type( t_unknown );
      std::cout << "Elementtype: " << arrayset->getArray_Type() << std::endl;
      xmlFree(str);

      // Parse shape
      size_t shape[4];
      shape[0]=shape[1]=shape[2]=shape[3]=0;
      str = xmlGetProp(cur, xmlCharStrdup(db::shape));
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
      arrayset->setN_dim(count);
      arrayset->setShape(shape);
      std::cout << "Nb dimensions: " << arrayset->getN_dim() << std::endl;
      std::cout << "Shape: (" << arrayset->getShape()[0] << "," << 
        arrayset->getShape()[1] << ","<< arrayset->getShape()[2] << "," << 
        arrayset->getShape()[3] << ")" << std::endl;
      xmlFree(str);
      // Set the number of elements
      size_t n_elem = arrayset->getShape()[0];
      for( size_t i=1; i < arrayset->getN_dim(); ++i)
        n_elem *= arrayset->getShape()[i];
      arrayset->setN_elem(n_elem);

      // Parse loader
      str = xmlGetProp(cur, xmlCharStrdup(db::loader));
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
      str = xmlGetProp(cur, xmlCharStrdup(db::file));
      arrayset->setFilename( (str!=0?(const char*)str:"") );
      struct stat stFileInfo;
      if( arrayset->getFilename().compare("") && 
          stat( arrayset->getFilename().c_str(), &stFileInfo) ) 
        Torch::core::warn << "The file" << arrayset->getFilename() << 
          " was not found." << std::endl;
      std::cout << "File: " << arrayset->getFilename() << std::endl;
      xmlFree(str);

      if( !arrayset->getFilename().compare("") )
      {
        // Parse the data
        xmlNodePtr cur_data = cur->xmlChildrenNode;

        Array_Type a_type = arrayset->getArray_Type();
        while (cur_data != 0) { 
          // Process an array
          if ((!xmlStrcmp(cur_data->name, xmlCharStrdup(db::array)))) {
            arrayset->add_array( parseArray( arrayset, cur_data, a_type, 
              arrayset->getN_elem() ) );
          }
          cur_data = cur_data->next;
        }
      }

      return arrayset;
    }


    boost::shared_ptr<Array> XMLParser::parseArray(
      const boost::shared_ptr<Arrayset> parent, 
      const xmlNodePtr cur, Array_Type a_type, size_t nb_values) 
    {
      boost::shared_ptr<Array> array(new Array(parent));
      // Parse id
      xmlChar *str;
      str = xmlGetProp(cur, xmlCharStrdup(db::id));
      array->setId(str!=0? static_cast<size_t>(atoi((const char*)str)): 0);
      std::cout << "  Array Id: " << array->getId() << std::endl;
      xmlFree(str);

      // Parse loader
      str = xmlGetProp(cur, xmlCharStrdup(db::loader));
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
      str = xmlGetProp(cur, xmlCharStrdup(db::file));
      array->setFilename( (str!=0?(const char*)str:"") );
      struct stat stFileInfo;
      if( array->getFilename().compare("") && 
          stat( array->getFilename().c_str(), &stFileInfo) ) 
        Torch::core::warn << "The file" << array->getFilename() << 
          " was not found." << std::endl;
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

        // Switch over the possible type
        switch( a_type) {
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


    template <typename T> T* XMLParser::parseArrayData( 
      boost::tokenizer<boost::char_separator<char> > tok, size_t nb_values )
    {
      T* data_array = new T[nb_values];
      size_t count = 0;
      for( boost::tokenizer<boost::char_separator<char> >::iterator
          it=tok.begin(); it!=tok.end(); ++it, ++count ) 
      {
        data_array[count] = boost::lexical_cast<T>(*it);
        std::cout << data_array[count] << " ";
      }
      std::cout << std::endl;

      if(count != nb_values) {
        Torch::core::error << "The number of values read (" << count <<
          ") in the array does not match with the expected number (" << 
          nb_values << ")" << std::endl;
        throw Exception();
      }

      return data_array;
    }


  }
}

