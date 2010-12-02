/**
 * @file src/core/src/DatasetXML.cc
 * @author <a href="mailto:Laurent.El-Shafey@idiap.ch">Laurent El Shafey</a>
 *
 * @brief Implements the DatasetXML class.
 */

#include "core/DatasetXML.h"
#include "core/Exception.h"
#include <iostream>
#include <boost/tokenizer.hpp>

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
    }

    ArraysetXML::ArraysetXML(const xmlNodePtr& arrayset) {
      // Parse id
      xmlChar *str;
      str = xmlGetProp(arrayset, xmlCharStrdup(db::id));
      m_id = static_cast<size_t>(atoi( (const char*)str ));
      std::cout << "Id: " << m_id << std::endl;
      xmlFree(str);

      // Parse role
      str = xmlGetProp(arrayset, xmlCharStrdup(db::role));
      m_role.assign( (str!=0?(const char *)str:"") );
      std::cout << "Role: " << m_role << std::endl;
      xmlFree(str);

      // Parse elementtype
      str = xmlGetProp(arrayset, xmlCharStrdup(db::elementtype));
      if( str==0 ) {
        error << "Elementtype is not specified in arrayset (id: " << 
          m_id << ")." << std::endl;
        throw Exception();
      }
      std::cout << "Elementtype: " << m_elementtype << std::endl;
      xmlFree(str);

      // Parse shape
      m_shape[0]=m_shape[1]=m_shape[2]=m_shape[3]=0;
      str = xmlGetProp(arrayset, xmlCharStrdup(db::shape));
      if( str==0 ) {
        error << "Elementtype is not specified in arrayset (id: " << 
          m_id << ")." << std::endl;
        throw Exception();
      }
      // Tokenize the shape string to extract the dimensions
      std::string shape((const char *)str);
      boost::tokenizer<> tok(shape);
      size_t count=0;
      for( boost::tokenizer<>::iterator it=tok.begin(); it!=tok.end(); 
        ++it, ++count ) 
      {
        if(count>3) {
          error << "Shape is not valid in arrayset (id: " << 
            m_id << "). Maximum number of dimensions is 4." << std::endl;
          throw Exception();        
        }
        m_shape[count] = atoi((*it).c_str());
      }
      std::cout << "Shape: (" << m_shape[0] <<","<< m_shape[1] << ","<< 
        m_shape[2] << "," << m_shape[3] << ")" << std::endl;
      xmlFree(str);

      // Parse loader
      str = xmlGetProp(arrayset, xmlCharStrdup(db::loader));
      std::cout << "Loader: " << (str!=0?str:xmlCharStrdup("")) << std::endl;
      xmlFree(str);
     
      // File loader
      str = xmlGetProp(arrayset, xmlCharStrdup(db::file));
      m_filename.assign( (str!=0?(const char*)str:"") );
      std::cout << "File: " << m_filename << std::endl << std::endl;
      xmlFree(str);

      // TODO: parse/load the arrays
    }


    DatasetXML::DatasetXML(char *filename) {
      parseFile(filename);
    }


    void DatasetXML::parseFile(char *filename) {
      m_doc = xmlParseFile(filename);
      xmlNodePtr cur;

      // Check validity of the XML document
      if(m_doc == 0 ) {
        error << "Document " << filename << " was not parsed successfully." << std::endl;
        throw Exception();
      }
      TDEBUG3("Document has been parsed succesfully.\n");

      cur = xmlDocGetRootElement(m_doc);
      if (cur == 0) {
        error << "Document " << filename << " is empty." << std::endl;
        xmlFreeDoc(m_doc);
        throw Exception();
      }
      TDEBUG3("Document is not empty.\n");

      if (xmlStrcmp(cur->name, xmlCharStrdup(db::dataset))) {
        error << "Document " << filename << " is of the wrong type (!= dataset)." << std::endl;
        xmlFreeDoc(m_doc);
        throw Exception();
      }
      TDEBUG3("Document is of correct type (dataset).\n");

      // Parse Arraysets
      ArraysetXML* ar;
      cur = cur->xmlChildrenNode;
      while (cur != 0) {
        if ((!xmlStrcmp(cur->name, xmlCharStrdup(db::arrayset)))) {
          ar = new ArraysetXML(cur);
          m_arrayset.insert ( std::pair<size_t,const ArraysetXML*>(ar->getId(), ar) );
        }
        cur = cur->next;
      }
    }


    DatasetXML::~DatasetXML() {
      if(m_doc) xmlFreeDoc(m_doc);
      
      // Remove the arrayset
      std::map<size_t,const ArraysetXML*>::iterator it;
      for ( it=m_arrayset.begin() ; it != m_arrayset.end(); ++it ) {
        const ArraysetXML *ar=it->second;
        m_arrayset.erase(it);
        delete ar;
      }
    }

    Dataset::const_iterator DatasetXML::begin() const {
      DatasetXML::const_iteratorXML* it=new DatasetXML::const_iteratorXML();
      return *it;
    }

    Dataset::const_iterator DatasetXML::end() const {
      DatasetXML::const_iteratorXML* it=new DatasetXML::const_iteratorXML();
      return *it;
    }


    const Arrayset& DatasetXML::at( const size_t id ) const {
      // Check that an arrayset with the given id exists
      if(m_arrayset.find( id ) == m_arrayset.end() )
        throw Exception();
      // return the arrayset if ok
      else
        return *(m_arrayset.find(id)->second);
    }

  
  }
}

