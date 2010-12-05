/**
 * @file src/core/src/XMLParser.cc
 * @author <a href="mailto:Laurent.El-Shafey@idiap.ch">Laurent El Shafey</a>
 *
 * @brief Implements the XML parser for a dataset.
 */

#include "core/XMLParser.h"
#include "core/Exception.h"

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
    }



    XMLParser::XMLParser() { }


    XMLParser::~XMLParser() { }


    Dataset* XMLParser::load(const char* filename) {
      // Parse the XML file with libxml2
      m_doc = xmlParseFile(filename);
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

      Dataset* dataset=new Dataset();

      // Parse Arraysets
      cur = cur->xmlChildrenNode;
      while (cur != 0) { 
        // Parse an arrayset and add it to the dataset
        if ((!xmlStrcmp(cur->name, xmlCharStrdup(db::arrayset)))) 
          dataset->add_arrayset( parseArrayset(cur) );
        cur = cur->next;
      }

      return dataset;
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

      // TODO: 1/ parse filename and loader
      //       2/ stat the filename
      //       3/ check that there is no data (if a filename is given)
      //       4/ if not filename do the following

      // Parse the data
      xmlNodePtr cur_data = cur->xmlChildrenNode;
      while (cur_data != 0) { 
        // Process an array
        if ((!xmlStrcmp(cur_data->name, xmlCharStrdup(db::array)))) {
          arrayset->add_array( parseArray(cur_data) );
        }
        cur_data = cur_data->next;
      }

      return arrayset;
    }


    boost::shared_ptr<Array> XMLParser::parseArray(const xmlNodePtr cur) {
      boost::shared_ptr<Array> array(new Array());
      // Parse id
      xmlChar *str;
      str = xmlGetProp(cur, xmlCharStrdup(db::id));
      array->setId(str!=0? static_cast<size_t>(atoi((const char*)str)): 0);
      std::cout << "Id: " << array->getId() << std::endl;
      xmlFree(str);

      //TODO: parse other attributes inclusive data
      
      return array;
    }



  }
}

