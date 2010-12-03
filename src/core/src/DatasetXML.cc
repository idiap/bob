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


    ArraysetXML::ArraysetXML(const xmlNodePtr& arrayset): m_blitz_type(0) {
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
      m_elementtype.assign( (const char *)str );
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
      m_nb_dim = count;
      std::cout << "Nb dimensions: " << m_nb_dim << std::endl;
      std::cout << "Shape: (" << m_shape[0] <<","<< m_shape[1] << ","<< 
        m_shape[2] << "," << m_shape[3] << ")" << std::endl;
      xmlFree(str);

      // Update the 'blitz type' attribute
      if( !m_elementtype.compare(db::t_bool) ) {
        switch(m_nb_dim) {
          case 1: m_blitz_type = 1; break;
          case 2: m_blitz_type = 2; break;
          case 3: m_blitz_type = 3; break;
          case 4: m_blitz_type = 4; break;
        }
      }
      else if( !m_elementtype.compare(db::t_int8) ) {
        switch(m_nb_dim) {
          case 1: m_blitz_type = 5; break;
          case 2: m_blitz_type = 6; break;
          case 3: m_blitz_type = 7; break;
          case 4: m_blitz_type = 8; break;
        }
      }
      else if( !m_elementtype.compare(db::t_int16) ) {
        switch(m_nb_dim) {
          case 1: m_blitz_type = 9; break;
          case 2: m_blitz_type = 10; break;
          case 3: m_blitz_type = 11; break;
          case 4: m_blitz_type = 12; break;
        }
      }
      else if( !m_elementtype.compare(db::t_int32) ) {
        switch(m_nb_dim) {
          case 1: m_blitz_type = 13; break;
          case 2: m_blitz_type = 14; break;
          case 3: m_blitz_type = 15; break;
          case 4: m_blitz_type = 16; break;
        }
      }
      else if( !m_elementtype.compare(db::t_int64) ) {
        switch(m_nb_dim) {
          case 1: m_blitz_type = 17; break;
          case 2: m_blitz_type = 18; break;
          case 3: m_blitz_type = 19; break;
          case 4: m_blitz_type = 20; break;
        }
      }
      else if( !m_elementtype.compare(db::t_uint8) ) {
        switch(m_nb_dim) {
          case 1: m_blitz_type = 21; break;
          case 2: m_blitz_type = 22; break;
          case 3: m_blitz_type = 23; break;
          case 4: m_blitz_type = 24; break;
        }
      }
      else if( !m_elementtype.compare(db::t_uint16) ) {
        switch(m_nb_dim) {
          case 1: m_blitz_type = 25; break;
          case 2: m_blitz_type = 26; break;
          case 3: m_blitz_type = 27; break;
          case 4: m_blitz_type = 28; break;
        }
      }
      else if( !m_elementtype.compare(db::t_uint32) ) {
        switch(m_nb_dim) {
          case 1: m_blitz_type = 29; break;
          case 2: m_blitz_type = 30; break;
          case 3: m_blitz_type = 31; break;
          case 4: m_blitz_type = 32; break;
        }
      }
      else if( !m_elementtype.compare(db::t_uint64) ) {
        switch(m_nb_dim) {
          case 1: m_blitz_type = 33; break;
          case 2: m_blitz_type = 34; break;
          case 3: m_blitz_type = 35; break;
          case 4: m_blitz_type = 36; break;
        }
      }
      else if( !m_elementtype.compare(db::t_float32) ) {
        switch(m_nb_dim) {
          case 1: m_blitz_type = 37; break;
          case 2: m_blitz_type = 38; break;
          case 3: m_blitz_type = 39; break;
          case 4: m_blitz_type = 40; break;
        }
      }
      else if( !m_elementtype.compare(db::t_float64) ) {
        switch(m_nb_dim) {
          case 1: m_blitz_type = 41; break;
          case 2: m_blitz_type = 42; break;
          case 3: m_blitz_type = 43; break;
          case 4: m_blitz_type = 44; break;
        }
      }
      else if( !m_elementtype.compare(db::t_float128) ) {
        switch(m_nb_dim) {
          case 1: m_blitz_type = 45; break;
          case 2: m_blitz_type = 46; break;
          case 3: m_blitz_type = 47; break;
          case 4: m_blitz_type = 48; break;
        }
      }
      else if( !m_elementtype.compare(db::t_complex64) ) {
        switch(m_nb_dim) {
          case 1: m_blitz_type = 49; break;
          case 2: m_blitz_type = 50; break;
          case 3: m_blitz_type = 51; break;
          case 4: m_blitz_type = 52; break;
        }
      }
      else if( !m_elementtype.compare(db::t_complex128) ) {
        switch(m_nb_dim) {
          case 1: m_blitz_type = 53; break;
          case 2: m_blitz_type = 54; break;
          case 3: m_blitz_type = 55; break;
          case 4: m_blitz_type = 56; break;
        }
      }
      else if( !m_elementtype.compare(db::t_complex256) ) {
        switch(m_nb_dim) {
          case 1: m_blitz_type = 57; break;
          case 2: m_blitz_type = 58; break;
          case 3: m_blitz_type = 59; break;
          case 4: m_blitz_type = 60; break;
        }
      }
      std::cout << "Blitz_type: " << m_blitz_type << std::endl;

      // Parse loader
      str = xmlGetProp(arrayset, xmlCharStrdup(db::loader));
      m_loader.assign( (str!=0?(const char*)str:"") );
      std::cout << "Loader: " << m_loader << std::endl;
      xmlFree(str);
     
      // Parse file
      str = xmlGetProp(arrayset, xmlCharStrdup(db::file));
      m_filename.assign( (str!=0?(const char*)str:"") );
      std::cout << "File: " << m_filename << std::endl;
      xmlFree(str);

      // parse/load the arrays contained in the XML file
      if(!m_filename.compare(""))
      {
        std::cout << "No filename given. Reading the data..." << std::endl;
        xmlNodePtr cur = arrayset->xmlChildrenNode;
        // Loop over all the nodes
        while (cur != 0) {
          // Process an array
          if ((!xmlStrcmp(cur->name, xmlCharStrdup(db::array)))) {
            // Parse id
            str = xmlGetProp(cur, xmlCharStrdup(db::id));
            size_t cur_id = 
              static_cast<size_t>(str!=0 ? atoi((const char*)str) : 0);
            std::cout << "Id: " << cur_id << std::endl;
            xmlFree(str);

            // Parse loader
            str = xmlGetProp(cur, xmlCharStrdup(db::loader));
            std::string s_loader( (str!=0?(const char*)str:"") );
            std::cout << "Loader: " << s_loader << std::endl;
            xmlFree(str);

            // Parse file
            str = xmlGetProp(cur, xmlCharStrdup(db::file));
            std::string s_filename( (str!=0?(const char*)str:"") );
            std::cout << "File: " << s_filename << std::endl;
            xmlFree(str);

            if(!s_filename.compare(""))
            {
              // Preliminary for the processing of the content of the array
              xmlChar* content = xmlNodeGetContent(cur);
              std::string data( (const char *)content);
              boost::char_separator<char> sep(" ;|");
              boost::tokenizer<boost::char_separator<char> > tok(data, sep);
              size_t total = 1;
              for(size_t i=0; i<m_nb_dim; ++i)
                total *= m_shape[i];
        
              count = 0;
              // Insert a new blitz array in the map of the arrayset type
              // and fillt it in
              int i0,i1,i2,i3,tmp1,tmp2;
              switch(m_blitz_type)
              {
                case 1: m_data_bool_1.insert(
                          std::pair<size_t,blitz::Array<bool,1> >
                          ( cur_id, blitz::Array<bool,1>(m_shape[0]) ) 
                          );
                  for( boost::tokenizer<boost::char_separator<char> >::iterator
                    it=tok.begin(); it!=tok.end(); ++it, ++count ) 
                  {
                    m_data_bool_1.find(cur_id)->second(count) = boost::lexical_cast<bool>(*it);
                    std::cout << m_data_bool_1.find(cur_id)->second(count) << " ";
                  }
                  break;
                case 2: m_data_bool_2.insert(
                          std::pair<size_t,blitz::Array<bool,2> >
                          ( cur_id, blitz::Array<bool,2>(m_shape[0], m_shape[1]) ) 
                          );
                  for( boost::tokenizer<boost::char_separator<char> >::iterator
                    it=tok.begin(); it!=tok.end(); ++it, ++count ) 
                  {
                    i0=count / m_shape[1];
                    i1=count % m_shape[1];
                    m_data_bool_2.find(cur_id)->second(i0,i1) = boost::lexical_cast<bool>(*it);
                    std::cout << m_data_bool_2.find(cur_id)->second(i0,i1) << " ";
                  }
                  break;
                case 3: m_data_bool_3.insert(
                          std::pair<size_t,blitz::Array<bool,3> >
                          ( cur_id, blitz::Array<bool,3>(m_shape[0], m_shape[1], m_shape[2]) ) 
                          );
                  for( boost::tokenizer<boost::char_separator<char> >::iterator
                    it=tok.begin(); it!=tok.end(); ++it, ++count ) 
                  {
                    i0=count / (m_shape[1]*m_shape[2]);
                    tmp1=count - i0*(m_shape[1]*m_shape[2]);
                    i1=tmp1 / m_shape[2];
                    i2=tmp1 % m_shape[2];
                    m_data_bool_3.find(cur_id)->second(i0,i1,i2) = boost::lexical_cast<bool>(*it);
                    std::cout << m_data_bool_3.find(cur_id)->second(i0,i1,i2) << " ";
                  }
                  break;
                case 4: m_data_bool_4.insert(
                          std::pair<size_t,blitz::Array<bool,4> >
                          ( cur_id, blitz::Array<bool,4>(m_shape[0], m_shape[1], m_shape[2], m_shape[3]) ) 
                          );
                  for( boost::tokenizer<boost::char_separator<char> >::iterator
                    it=tok.begin(); it!=tok.end(); ++it, ++count ) 
                  {
                    i0=count / (m_shape[1]*m_shape[2]*m_shape[3]);
                    tmp1=count - i0*(m_shape[1]*m_shape[2]*m_shape[3]);
                    i1=tmp1 / (m_shape[2]*m_shape[3]);
                    tmp2=tmp1 - i1*(m_shape[2]*m_shape[3]);
                    i2=tmp2 / m_shape[3];
                    i3=tmp2 % m_shape[3];
                    m_data_bool_4.find(cur_id)->second(i0,i1,i2,i3) = boost::lexical_cast<bool>(*it);
                    std::cout << m_data_bool_4.find(cur_id)->second(i0,i1,i2,i3) << " ";
                  }
                  break;
                case 5: m_data_int8_1.insert(
                          std::pair<size_t,blitz::Array<int8_t,1> >
                          ( cur_id, blitz::Array<int8_t,1>(m_shape[0]) ) 
                          );
                  for( boost::tokenizer<boost::char_separator<char> >::iterator
                    it=tok.begin(); it!=tok.end(); ++it, ++count ) 
                  {
                    m_data_int8_1.find(cur_id)->second(count) = boost::lexical_cast<int8_t>(*it);
                    std::cout << m_data_int8_1.find(cur_id)->second(count) << " ";
                  }
                  break;
                case 6: m_data_int8_2.insert(
                          std::pair<size_t,blitz::Array<int8_t,2> >
                          ( cur_id, blitz::Array<int8_t,2>(m_shape[0], m_shape[1]) ) 
                          );
                  for( boost::tokenizer<boost::char_separator<char> >::iterator
                    it=tok.begin(); it!=tok.end(); ++it, ++count ) 
                  {
                    i0=count / m_shape[1];
                    i1=count % m_shape[1];
                    m_data_int8_2.find(cur_id)->second(i0,i1) = boost::lexical_cast<int8_t>(*it);
                    std::cout << m_data_int8_2.find(cur_id)->second(i0,i1) << " ";
                  }
                  break;
                case 7: m_data_int8_3.insert(
                          std::pair<size_t,blitz::Array<int8_t,3> >
                          ( cur_id, blitz::Array<int8_t,3>(m_shape[0], m_shape[1], m_shape[2]) ) 
                          );
                  for( boost::tokenizer<boost::char_separator<char> >::iterator
                    it=tok.begin(); it!=tok.end(); ++it, ++count ) 
                  {
                    i0=count / (m_shape[1]*m_shape[2]);
                    tmp1=count - i0*(m_shape[1]*m_shape[2]);
                    i1=tmp1 / m_shape[2];
                    i2=tmp1 % m_shape[2];
                    m_data_int8_3.find(cur_id)->second(i0,i1,i2) = boost::lexical_cast<int8_t>(*it);
                    std::cout << m_data_int8_3.find(cur_id)->second(i0,i1,i2) << " ";
                  }
                  break;
                case 8: m_data_int8_4.insert(
                          std::pair<size_t,blitz::Array<int8_t,4> >
                          ( cur_id, blitz::Array<int8_t,4>(m_shape[0], m_shape[1], m_shape[2], m_shape[3]) ) 
                          );
                  for( boost::tokenizer<boost::char_separator<char> >::iterator
                    it=tok.begin(); it!=tok.end(); ++it, ++count ) 
                  {
                    i0=count / (m_shape[1]*m_shape[2]*m_shape[3]);
                    tmp1=count - i0*(m_shape[1]*m_shape[2]*m_shape[3]);
                    i1=tmp1 / (m_shape[2]*m_shape[3]);
                    tmp2=tmp1 - i1*(m_shape[2]*m_shape[3]);
                    i2=tmp2 / m_shape[3];
                    i3=tmp2 % m_shape[3];
                    m_data_int8_4.find(cur_id)->second(i0,i1,i2,i3) = boost::lexical_cast<int8_t>(*it);
                    std::cout << m_data_int8_4.find(cur_id)->second(i0,i1,i2,i3) << " ";
                  }
                  break;
                case 9: m_data_int16_1.insert(
                          std::pair<size_t,blitz::Array<int16_t,1> >
                          ( cur_id, blitz::Array<int16_t,1>(m_shape[0]) ) 
                          );
                  for( boost::tokenizer<boost::char_separator<char> >::iterator
                    it=tok.begin(); it!=tok.end(); ++it, ++count ) 
                  {
                    m_data_int16_1.find(cur_id)->second(count) = boost::lexical_cast<int16_t>(*it);
                    std::cout << m_data_int16_1.find(cur_id)->second(count) << " ";
                  }
                  break;
                case 10: m_data_int16_2.insert(
                          std::pair<size_t,blitz::Array<int16_t,2> >
                          ( cur_id, blitz::Array<int16_t,2>(m_shape[0], m_shape[1]) ) 
                          );
                  for( boost::tokenizer<boost::char_separator<char> >::iterator
                    it=tok.begin(); it!=tok.end(); ++it, ++count ) 
                  {
                    i0=count / m_shape[1];
                    i1=count % m_shape[1];
                    m_data_int16_2.find(cur_id)->second(i0,i1) = boost::lexical_cast<int16_t>(*it);
                    std::cout << m_data_int16_2.find(cur_id)->second(i0,i1) << " ";
                  }
                  break;
                case 11: m_data_int16_3.insert(
                          std::pair<size_t,blitz::Array<int16_t,3> >
                          ( cur_id, blitz::Array<int16_t,3>(m_shape[0], m_shape[1], m_shape[2]) ) 
                          );
                  for( boost::tokenizer<boost::char_separator<char> >::iterator
                    it=tok.begin(); it!=tok.end(); ++it, ++count ) 
                  {
                    i0=count / (m_shape[1]*m_shape[2]);
                    tmp1=count - i0*(m_shape[1]*m_shape[2]);
                    i1=tmp1 / m_shape[2];
                    i2=tmp1 % m_shape[2];
                    m_data_int16_3.find(cur_id)->second(i0,i1,i2) = boost::lexical_cast<int16_t>(*it);
                    std::cout << m_data_int16_3.find(cur_id)->second(i0,i1,i2) << " ";
                  }
                  break;
                case 12: m_data_int16_4.insert(
                          std::pair<size_t,blitz::Array<int16_t,4> >
                          ( cur_id, blitz::Array<int16_t,4>(m_shape[0], m_shape[1], m_shape[2], m_shape[3]) ) 
                          );
                  for( boost::tokenizer<boost::char_separator<char> >::iterator
                    it=tok.begin(); it!=tok.end(); ++it, ++count ) 
                  {
                    i0=count / (m_shape[1]*m_shape[2]*m_shape[3]);
                    tmp1=count - i0*(m_shape[1]*m_shape[2]*m_shape[3]);
                    i1=tmp1 / (m_shape[2]*m_shape[3]);
                    tmp2=tmp1 - i1*(m_shape[2]*m_shape[3]);
                    i2=tmp2 / m_shape[3];
                    i3=tmp2 % m_shape[3];
                    m_data_int16_4.find(cur_id)->second(i0,i1,i2,i3) = boost::lexical_cast<int16_t>(*it);
                    std::cout << m_data_int16_4.find(cur_id)->second(i0,i1,i2,i3) << " ";
                  }
                  break;
                case 13: m_data_int32_1.insert(
                          std::pair<size_t,blitz::Array<int32_t,1> >
                          ( cur_id, blitz::Array<int32_t,1>(m_shape[0]) ) 
                          );
                  for( boost::tokenizer<boost::char_separator<char> >::iterator
                    it=tok.begin(); it!=tok.end(); ++it, ++count ) 
                  {
                    m_data_int32_1.find(cur_id)->second(count) = boost::lexical_cast<int32_t>(*it);
                    std::cout << m_data_int32_1.find(cur_id)->second(count) << " ";
                  }
                  break;
                case 14: m_data_int32_2.insert(
                          std::pair<size_t,blitz::Array<int32_t,2> >
                          ( cur_id, blitz::Array<int32_t,2>(m_shape[0], m_shape[1]) ) 
                          );
                  for( boost::tokenizer<boost::char_separator<char> >::iterator
                    it=tok.begin(); it!=tok.end(); ++it, ++count ) 
                  {
                    i0=count / m_shape[1];
                    i1=count % m_shape[1];
                    m_data_int32_2.find(cur_id)->second(i0,i1) = boost::lexical_cast<int32_t>(*it);
                    std::cout << m_data_int32_2.find(cur_id)->second(i0,i1) << " ";
                  }
                  break;
                case 15: m_data_int32_3.insert(
                          std::pair<size_t,blitz::Array<int32_t,3> >
                          ( cur_id, blitz::Array<int32_t,3>(m_shape[0], m_shape[1], m_shape[2]) ) 
                          );
                  for( boost::tokenizer<boost::char_separator<char> >::iterator
                    it=tok.begin(); it!=tok.end(); ++it, ++count ) 
                  {
                    i0=count / (m_shape[1]*m_shape[2]);
                    tmp1=count - i0*(m_shape[1]*m_shape[2]);
                    i1=tmp1 / m_shape[2];
                    i2=tmp1 % m_shape[2];
                    m_data_int32_3.find(cur_id)->second(i0,i1,i2) = boost::lexical_cast<int32_t>(*it);
                    std::cout << m_data_int32_3.find(cur_id)->second(i0,i1,i2) << " ";
                  }
                  break;
                case 16: m_data_int32_4.insert(
                          std::pair<size_t,blitz::Array<int32_t,4> >
                          ( cur_id, blitz::Array<int32_t,4>(m_shape[0], m_shape[1], m_shape[2], m_shape[3]) ) 
                          );
                  for( boost::tokenizer<boost::char_separator<char> >::iterator
                    it=tok.begin(); it!=tok.end(); ++it, ++count ) 
                  {
                    i0=count / (m_shape[1]*m_shape[2]*m_shape[3]);
                    tmp1=count - i0*(m_shape[1]*m_shape[2]*m_shape[3]);
                    i1=tmp1 / (m_shape[2]*m_shape[3]);
                    tmp2=tmp1 - i1*(m_shape[2]*m_shape[3]);
                    i2=tmp2 / m_shape[3];
                    i3=tmp2 % m_shape[3];
                    m_data_int32_4.find(cur_id)->second(i0,i1,i2,i3) = boost::lexical_cast<int32_t>(*it);
                    std::cout << m_data_int32_4.find(cur_id)->second(i0,i1,i2,i3) << " ";
                  }
                  break;
                case 17: m_data_int64_1.insert(
                          std::pair<size_t,blitz::Array<int64_t,1> >
                          ( cur_id, blitz::Array<int64_t,1>(m_shape[0]) ) 
                          );
                  for( boost::tokenizer<boost::char_separator<char> >::iterator
                    it=tok.begin(); it!=tok.end(); ++it, ++count ) 
                  {
                    m_data_int64_1.find(cur_id)->second(count) = boost::lexical_cast<int64_t>(*it);
                    std::cout << m_data_int64_1.find(cur_id)->second(count) << " ";
                  }
                  break;
                case 18: m_data_int64_2.insert(
                          std::pair<size_t,blitz::Array<int64_t,2> >
                          ( cur_id, blitz::Array<int64_t,2>(m_shape[0], m_shape[1]) ) 
                          );
                  for( boost::tokenizer<boost::char_separator<char> >::iterator
                    it=tok.begin(); it!=tok.end(); ++it, ++count ) 
                  {
                    i0=count / m_shape[1];
                    i1=count % m_shape[1];
                    m_data_int64_2.find(cur_id)->second(i0,i1) = boost::lexical_cast<int64_t>(*it);
                    std::cout << m_data_int64_2.find(cur_id)->second(i0,i1) << " ";
                  }
                  break;
                case 19: m_data_int64_3.insert(
                          std::pair<size_t,blitz::Array<int64_t,3> >
                          ( cur_id, blitz::Array<int64_t,3>(m_shape[0], m_shape[1], m_shape[2]) ) 
                          );
                  for( boost::tokenizer<boost::char_separator<char> >::iterator
                    it=tok.begin(); it!=tok.end(); ++it, ++count ) 
                  {
                    i0=count / (m_shape[1]*m_shape[2]);
                    tmp1=count - i0*(m_shape[1]*m_shape[2]);
                    i1=tmp1 / m_shape[2];
                    i2=tmp1 % m_shape[2];
                    m_data_int64_3.find(cur_id)->second(i0,i1,i2) = boost::lexical_cast<int64_t>(*it);
                    std::cout << m_data_int64_3.find(cur_id)->second(i0,i1,i2) << " ";
                  }
                  break;
                case 20: m_data_int64_4.insert(
                          std::pair<size_t,blitz::Array<int64_t,4> >
                          ( cur_id, blitz::Array<int64_t,4>(m_shape[0], m_shape[1], m_shape[2], m_shape[3]) ) 
                          );
                  for( boost::tokenizer<boost::char_separator<char> >::iterator
                    it=tok.begin(); it!=tok.end(); ++it, ++count ) 
                  {
                    i0=count / (m_shape[1]*m_shape[2]*m_shape[3]);
                    tmp1=count - i0*(m_shape[1]*m_shape[2]*m_shape[3]);
                    i1=tmp1 / (m_shape[2]*m_shape[3]);
                    tmp2=tmp1 - i1*(m_shape[2]*m_shape[3]);
                    i2=tmp2 / m_shape[3];
                    i3=tmp2 % m_shape[3];
                    m_data_int64_4.find(cur_id)->second(i0,i1,i2,i3) = boost::lexical_cast<int64_t>(*it);
                    std::cout << m_data_int64_4.find(cur_id)->second(i0,i1,i2,i3) << " ";
                  }
                  break;
                case 21: m_data_uint8_1.insert(
                          std::pair<size_t,blitz::Array<uint8_t,1> >
                          ( cur_id, blitz::Array<uint8_t,1>(m_shape[0]) ) 
                          );
                  for( boost::tokenizer<boost::char_separator<char> >::iterator
                    it=tok.begin(); it!=tok.end(); ++it, ++count ) 
                  {
                    m_data_uint8_1.find(cur_id)->second(count) = boost::lexical_cast<uint8_t>(*it);
                    std::cout << m_data_uint8_1.find(cur_id)->second(count) << " ";
                  }
                  break;
                case 22: m_data_uint8_2.insert(
                          std::pair<size_t,blitz::Array<uint8_t,2> >
                          ( cur_id, blitz::Array<uint8_t,2>(m_shape[0], m_shape[1]) ) 
                          );
                  for( boost::tokenizer<boost::char_separator<char> >::iterator
                    it=tok.begin(); it!=tok.end(); ++it, ++count ) 
                  {
                    i0=count / m_shape[1];
                    i1=count % m_shape[1];
                    m_data_uint8_2.find(cur_id)->second(i0,i1) = boost::lexical_cast<uint8_t>(*it);
                    std::cout << m_data_uint8_2.find(cur_id)->second(i0,i1) << " ";
                  }
                  break;
                case 23: m_data_uint8_3.insert(
                          std::pair<size_t,blitz::Array<uint8_t,3> >
                          ( cur_id, blitz::Array<uint8_t,3>(m_shape[0], m_shape[1], m_shape[2]) ) 
                          );
                  for( boost::tokenizer<boost::char_separator<char> >::iterator
                    it=tok.begin(); it!=tok.end(); ++it, ++count ) 
                  {
                    i0=count / (m_shape[1]*m_shape[2]);
                    tmp1=count - i0*(m_shape[1]*m_shape[2]);
                    i1=tmp1 / m_shape[2];
                    i2=tmp1 % m_shape[2];
                    m_data_uint8_3.find(cur_id)->second(i0,i1,i2) = boost::lexical_cast<uint8_t>(*it);
                    std::cout << m_data_uint8_3.find(cur_id)->second(i0,i1,i2) << " ";
                  }
                  break;
                case 24: m_data_uint8_4.insert(
                          std::pair<size_t,blitz::Array<uint8_t,4> >
                          ( cur_id, blitz::Array<uint8_t,4>(m_shape[0], m_shape[1], m_shape[2], m_shape[3]) ) 
                          );
                  for( boost::tokenizer<boost::char_separator<char> >::iterator
                    it=tok.begin(); it!=tok.end(); ++it, ++count ) 
                  {
                    i0=count / (m_shape[1]*m_shape[2]*m_shape[3]);
                    tmp1=count - i0*(m_shape[1]*m_shape[2]*m_shape[3]);
                    i1=tmp1 / (m_shape[2]*m_shape[3]);
                    tmp2=tmp1 - i1*(m_shape[2]*m_shape[3]);
                    i2=tmp2 / m_shape[3];
                    i3=tmp2 % m_shape[3];
                    m_data_uint8_4.find(cur_id)->second(i0,i1,i2,i3) = boost::lexical_cast<uint8_t>(*it);
                    std::cout << m_data_uint8_4.find(cur_id)->second(i0,i1,i2,i3) << " ";
                  }
                  break;
                case 25: m_data_uint16_1.insert(
                          std::pair<size_t,blitz::Array<uint16_t,1> >
                          ( cur_id, blitz::Array<uint16_t,1>(m_shape[0]) ) 
                          );
                  for( boost::tokenizer<boost::char_separator<char> >::iterator
                    it=tok.begin(); it!=tok.end(); ++it, ++count ) 
                  {
                    m_data_uint16_1.find(cur_id)->second(count) = boost::lexical_cast<uint16_t>(*it);
                    std::cout << m_data_uint16_1.find(cur_id)->second(count) << " ";
                  }
                  break;
                case 26: m_data_uint16_2.insert(
                          std::pair<size_t,blitz::Array<uint16_t,2> >
                          ( cur_id, blitz::Array<uint16_t,2>(m_shape[0], m_shape[1]) ) 
                          );
                  for( boost::tokenizer<boost::char_separator<char> >::iterator
                    it=tok.begin(); it!=tok.end(); ++it, ++count ) 
                  {
                    i0=count / m_shape[1];
                    i1=count % m_shape[1];
                    m_data_uint16_2.find(cur_id)->second(i0,i1) = boost::lexical_cast<uint16_t>(*it);
                    std::cout << m_data_uint16_2.find(cur_id)->second(i0,i1) << " ";
                  }
                  break;
                case 27: m_data_uint16_3.insert(
                          std::pair<size_t,blitz::Array<uint16_t,3> >
                          ( cur_id, blitz::Array<uint16_t,3>(m_shape[0], m_shape[1], m_shape[2]) ) 
                          );
                  for( boost::tokenizer<boost::char_separator<char> >::iterator
                    it=tok.begin(); it!=tok.end(); ++it, ++count ) 
                  {
                    i0=count / (m_shape[1]*m_shape[2]);
                    tmp1=count - i0*(m_shape[1]*m_shape[2]);
                    i1=tmp1 / m_shape[2];
                    i2=tmp1 % m_shape[2];
                    m_data_uint16_3.find(cur_id)->second(i0,i1,i2) = boost::lexical_cast<uint16_t>(*it);
                    std::cout << m_data_uint16_3.find(cur_id)->second(i0,i1,i2) << " ";
                  }
                  break;
                case 28: m_data_uint16_4.insert(
                          std::pair<size_t,blitz::Array<uint16_t,4> >
                          ( cur_id, blitz::Array<uint16_t,4>(m_shape[0], m_shape[1], m_shape[2], m_shape[3]) ) 
                          );
                  for( boost::tokenizer<boost::char_separator<char> >::iterator
                    it=tok.begin(); it!=tok.end(); ++it, ++count ) 
                  {
                    i0=count / (m_shape[1]*m_shape[2]*m_shape[3]);
                    tmp1=count - i0*(m_shape[1]*m_shape[2]*m_shape[3]);
                    i1=tmp1 / (m_shape[2]*m_shape[3]);
                    tmp2=tmp1 - i1*(m_shape[2]*m_shape[3]);
                    i2=tmp2 / m_shape[3];
                    i3=tmp2 % m_shape[3];
                    m_data_uint16_4.find(cur_id)->second(i0,i1,i2,i3) = boost::lexical_cast<uint16_t>(*it);
                    std::cout << m_data_uint16_4.find(cur_id)->second(i0,i1,i2,i3) << " ";
                  }
                  break;
                case 29: m_data_uint32_1.insert(
                          std::pair<size_t,blitz::Array<uint32_t,1> >
                          ( cur_id, blitz::Array<uint32_t,1>(m_shape[0]) ) 
                          );
                  for( boost::tokenizer<boost::char_separator<char> >::iterator
                    it=tok.begin(); it!=tok.end(); ++it, ++count ) 
                  {
                    m_data_uint32_1.find(cur_id)->second(count) = boost::lexical_cast<uint32_t>(*it);
                    std::cout << m_data_uint32_1.find(cur_id)->second(count) << " ";
                  }
                  break;
                case 30: m_data_uint32_2.insert(
                          std::pair<size_t,blitz::Array<uint32_t,2> >
                          ( cur_id, blitz::Array<uint32_t,2>(m_shape[0], m_shape[1]) ) 
                          );
                  for( boost::tokenizer<boost::char_separator<char> >::iterator
                    it=tok.begin(); it!=tok.end(); ++it, ++count ) 
                  {
                    i0=count / m_shape[1];
                    i1=count % m_shape[1];
                    m_data_uint32_2.find(cur_id)->second(i0,i1) = boost::lexical_cast<uint32_t>(*it);
                    std::cout << m_data_uint32_2.find(cur_id)->second(i0,i1) << " ";
                  }
                  break;
                case 31: m_data_uint32_3.insert(
                          std::pair<size_t,blitz::Array<uint32_t,3> >
                          ( cur_id, blitz::Array<uint32_t,3>(m_shape[0], m_shape[1], m_shape[2]) ) 
                          );
                  for( boost::tokenizer<boost::char_separator<char> >::iterator
                    it=tok.begin(); it!=tok.end(); ++it, ++count ) 
                  {
                    i0=count / (m_shape[1]*m_shape[2]);
                    tmp1=count - i0*(m_shape[1]*m_shape[2]);
                    i1=tmp1 / m_shape[2];
                    i2=tmp1 % m_shape[2];
                    m_data_uint32_3.find(cur_id)->second(i0,i1,i2) = boost::lexical_cast<uint32_t>(*it);
                    std::cout << m_data_uint32_3.find(cur_id)->second(i0,i1,i2) << " ";
                  }
                  break;
                case 32: m_data_uint32_4.insert(
                          std::pair<size_t,blitz::Array<uint32_t,4> >
                          ( cur_id, blitz::Array<uint32_t,4>(m_shape[0], m_shape[1], m_shape[2], m_shape[3]) ) 
                          );
                  for( boost::tokenizer<boost::char_separator<char> >::iterator
                    it=tok.begin(); it!=tok.end(); ++it, ++count ) 
                  {
                    i0=count / (m_shape[1]*m_shape[2]*m_shape[3]);
                    tmp1=count - i0*(m_shape[1]*m_shape[2]*m_shape[3]);
                    i1=tmp1 / (m_shape[2]*m_shape[3]);
                    tmp2=tmp1 - i1*(m_shape[2]*m_shape[3]);
                    i2=tmp2 / m_shape[3];
                    i3=tmp2 % m_shape[3];
                    m_data_uint32_4.find(cur_id)->second(i0,i1,i2,i3) = boost::lexical_cast<uint32_t>(*it);
                    std::cout << m_data_uint32_4.find(cur_id)->second(i0,i1,i2,i3) << " ";
                  }
                  break;
                case 33: m_data_uint64_1.insert(
                          std::pair<size_t,blitz::Array<uint64_t,1> >
                          ( cur_id, blitz::Array<uint64_t,1>(m_shape[0]) ) 
                          );
                  for( boost::tokenizer<boost::char_separator<char> >::iterator
                    it=tok.begin(); it!=tok.end(); ++it, ++count ) 
                  {
                    m_data_uint64_1.find(cur_id)->second(count) = boost::lexical_cast<uint64_t>(*it);
                    std::cout << m_data_uint64_1.find(cur_id)->second(count) << " ";
                  }
                  break;
                case 34: m_data_uint64_2.insert(
                          std::pair<size_t,blitz::Array<uint64_t,2> >
                          ( cur_id, blitz::Array<uint64_t,2>(m_shape[0], m_shape[1]) ) 
                          );
                  for( boost::tokenizer<boost::char_separator<char> >::iterator
                    it=tok.begin(); it!=tok.end(); ++it, ++count ) 
                  {
                    i0=count / m_shape[1];
                    i1=count % m_shape[1];
                    m_data_uint64_2.find(cur_id)->second(i0,i1) = boost::lexical_cast<uint64_t>(*it);
                    std::cout << m_data_uint64_2.find(cur_id)->second(i0,i1) << " ";
                  }
                  break;
                case 35: m_data_uint64_3.insert(
                          std::pair<size_t,blitz::Array<uint64_t,3> >
                          ( cur_id, blitz::Array<uint64_t,3>(m_shape[0], m_shape[1], m_shape[2]) ) 
                          );
                  for( boost::tokenizer<boost::char_separator<char> >::iterator
                    it=tok.begin(); it!=tok.end(); ++it, ++count ) 
                  {
                    i0=count / (m_shape[1]*m_shape[2]);
                    tmp1=count - i0*(m_shape[1]*m_shape[2]);
                    i1=tmp1 / m_shape[2];
                    i2=tmp1 % m_shape[2];
                    m_data_uint64_3.find(cur_id)->second(i0,i1,i2) = boost::lexical_cast<uint64_t>(*it);
                    std::cout << m_data_uint64_3.find(cur_id)->second(i0,i1,i2) << " ";
                  }
                  break;
                case 36: m_data_uint64_4.insert(
                          std::pair<size_t,blitz::Array<uint64_t,4> >
                          ( cur_id, blitz::Array<uint64_t,4>(m_shape[0], m_shape[1], m_shape[2], m_shape[3]) ) 
                          );
                  for( boost::tokenizer<boost::char_separator<char> >::iterator
                    it=tok.begin(); it!=tok.end(); ++it, ++count ) 
                  {
                    i0=count / (m_shape[1]*m_shape[2]*m_shape[3]);
                    tmp1=count - i0*(m_shape[1]*m_shape[2]*m_shape[3]);
                    i1=tmp1 / (m_shape[2]*m_shape[3]);
                    tmp2=tmp1 - i1*(m_shape[2]*m_shape[3]);
                    i2=tmp2 / m_shape[3];
                    i3=tmp2 % m_shape[3];
                    m_data_uint64_4.find(cur_id)->second(i0,i1,i2,i3) = boost::lexical_cast<uint64_t>(*it);
                    std::cout << m_data_uint64_4.find(cur_id)->second(i0,i1,i2,i3) << " ";
                  }
                  break;
                case 37: m_data_float32_1.insert(
                          std::pair<size_t,blitz::Array<float,1> >
                          ( cur_id, blitz::Array<float,1>(m_shape[0]) ) 
                          );
                  for( boost::tokenizer<boost::char_separator<char> >::iterator
                    it=tok.begin(); it!=tok.end(); ++it, ++count ) 
                  {
                    m_data_float32_1.find(cur_id)->second(count) = boost::lexical_cast<float>(*it);
                    std::cout << m_data_float32_1.find(cur_id)->second(count) << " ";
                  }
                  break;
                case 38: m_data_float32_2.insert(
                          std::pair<size_t,blitz::Array<float,2> >
                          ( cur_id, blitz::Array<float,2>(m_shape[0], m_shape[1]) ) 
                          );
                  for( boost::tokenizer<boost::char_separator<char> >::iterator
                    it=tok.begin(); it!=tok.end(); ++it, ++count ) 
                  {
                    i0=count / m_shape[1];
                    i1=count % m_shape[1];
                    m_data_float32_2.find(cur_id)->second(i0,i1) = boost::lexical_cast<float>(*it);
                    std::cout << m_data_float32_2.find(cur_id)->second(i0,i1) << " ";
                  }
                  break;
                case 39: m_data_float32_3.insert(
                          std::pair<size_t,blitz::Array<float,3> >
                          ( cur_id, blitz::Array<float,3>(m_shape[0], m_shape[1], m_shape[2]) ) 
                          );
                  for( boost::tokenizer<boost::char_separator<char> >::iterator
                    it=tok.begin(); it!=tok.end(); ++it, ++count ) 
                  {
                    i0=count / (m_shape[1]*m_shape[2]);
                    tmp1=count - i0*(m_shape[1]*m_shape[2]);
                    i1=tmp1 / m_shape[2];
                    i2=tmp1 % m_shape[2];
                    m_data_float32_3.find(cur_id)->second(i0,i1,i2) = boost::lexical_cast<float>(*it);
                    std::cout << m_data_float32_3.find(cur_id)->second(i0,i1,i2) << " ";
                  }
                  break;
                case 40: m_data_float32_4.insert(
                          std::pair<size_t,blitz::Array<float,4> >
                          ( cur_id, blitz::Array<float,4>(m_shape[0], m_shape[1], m_shape[2], m_shape[3]) ) 
                          );
                  for( boost::tokenizer<boost::char_separator<char> >::iterator
                    it=tok.begin(); it!=tok.end(); ++it, ++count ) 
                  {
                    i0=count / (m_shape[1]*m_shape[2]*m_shape[3]);
                    tmp1=count - i0*(m_shape[1]*m_shape[2]*m_shape[3]);
                    i1=tmp1 / (m_shape[2]*m_shape[3]);
                    tmp2=tmp1 - i1*(m_shape[2]*m_shape[3]);
                    i2=tmp2 / m_shape[3];
                    i3=tmp2 % m_shape[3];
                    m_data_float32_4.find(cur_id)->second(i0,i1,i2,i3) = boost::lexical_cast<float>(*it);
                    std::cout << m_data_float32_4.find(cur_id)->second(i0,i1,i2,i3) << " ";
                  }
                  break;
                case 41: m_data_float64_1.insert(
                          std::pair<size_t,blitz::Array<double,1> >
                          ( cur_id, blitz::Array<double,1>(m_shape[0]) ) 
                          );
                  for( boost::tokenizer<boost::char_separator<char> >::iterator
                    it=tok.begin(); it!=tok.end(); ++it, ++count ) 
                  {
                    m_data_float64_1.find(cur_id)->second(count) = boost::lexical_cast<double>(*it);
                    std::cout << m_data_float64_1.find(cur_id)->second(count) << " ";
                  }
                  break;
                case 42: m_data_float64_2.insert(
                          std::pair<size_t,blitz::Array<double,2> >
                          ( cur_id, blitz::Array<double,2>(m_shape[0], m_shape[1]) ) 
                          );
                  for( boost::tokenizer<boost::char_separator<char> >::iterator
                    it=tok.begin(); it!=tok.end(); ++it, ++count ) 
                  {
                    i0=count / m_shape[1];
                    i1=count % m_shape[1];
                    m_data_float64_2.find(cur_id)->second(i0,i1) = boost::lexical_cast<double>(*it);
                    std::cout << m_data_float64_2.find(cur_id)->second(i0,i1) << " ";
                  }
                  break;
                case 43: m_data_float64_3.insert(
                          std::pair<size_t,blitz::Array<double,3> >
                          ( cur_id, blitz::Array<double,3>(m_shape[0], m_shape[1], m_shape[2]) ) 
                          );
                  for( boost::tokenizer<boost::char_separator<char> >::iterator
                    it=tok.begin(); it!=tok.end(); ++it, ++count ) 
                  {
                    i0=count / (m_shape[1]*m_shape[2]);
                    tmp1=count - i0*(m_shape[1]*m_shape[2]);
                    i1=tmp1 / m_shape[2];
                    i2=tmp1 % m_shape[2];
                    m_data_float64_3.find(cur_id)->second(i0,i1,i2) = boost::lexical_cast<double>(*it);
                    std::cout << m_data_float64_3.find(cur_id)->second(i0,i1,i2) << " ";
                  }
                  break;
                case 44: m_data_float64_4.insert(
                          std::pair<size_t,blitz::Array<double,4> >
                          ( cur_id, blitz::Array<double,4>(m_shape[0], m_shape[1], m_shape[2], m_shape[3]) ) 
                          );
                  for( boost::tokenizer<boost::char_separator<char> >::iterator
                    it=tok.begin(); it!=tok.end(); ++it, ++count ) 
                  {
                    i0=count / (m_shape[1]*m_shape[2]*m_shape[3]);
                    tmp1=count - i0*(m_shape[1]*m_shape[2]*m_shape[3]);
                    i1=tmp1 / (m_shape[2]*m_shape[3]);
                    tmp2=tmp1 - i1*(m_shape[2]*m_shape[3]);
                    i2=tmp2 / m_shape[3];
                    i3=tmp2 % m_shape[3];
                    m_data_float64_4.find(cur_id)->second(i0,i1,i2,i3) = boost::lexical_cast<double>(*it);
                    std::cout << m_data_float64_4.find(cur_id)->second(i0,i1,i2,i3) << " ";
                  }
                  break;
                case 45: m_data_float128_1.insert(
                          std::pair<size_t,blitz::Array<long double,1> >
                          ( cur_id, blitz::Array<long double,1>(m_shape[0]) ) 
                          );
                  for( boost::tokenizer<boost::char_separator<char> >::iterator
                    it=tok.begin(); it!=tok.end(); ++it, ++count ) 
                  {
                    m_data_float128_1.find(cur_id)->second(count) = boost::lexical_cast<long double>(*it);
                    std::cout << m_data_float128_1.find(cur_id)->second(count) << " ";
                  }
                  break;
                case 46: m_data_float128_2.insert(
                          std::pair<size_t,blitz::Array<long double,2> >
                          ( cur_id, blitz::Array<long double,2>(m_shape[0], m_shape[1]) ) 
                          );
                  for( boost::tokenizer<boost::char_separator<char> >::iterator
                    it=tok.begin(); it!=tok.end(); ++it, ++count ) 
                  {
                    i0=count / m_shape[1];
                    i1=count % m_shape[1];
                    m_data_float128_2.find(cur_id)->second(i0,i1) = boost::lexical_cast<long double>(*it);
                    std::cout << m_data_float128_2.find(cur_id)->second(i0,i1) << " ";
                  }
                  break;
                case 47: m_data_float128_3.insert(
                          std::pair<size_t,blitz::Array<long double,3> >
                          ( cur_id, blitz::Array<long double,3>(m_shape[0], m_shape[1], m_shape[2]) ) 
                          );
                  for( boost::tokenizer<boost::char_separator<char> >::iterator
                    it=tok.begin(); it!=tok.end(); ++it, ++count ) 
                  {
                    i0=count / (m_shape[1]*m_shape[2]);
                    tmp1=count - i0*(m_shape[1]*m_shape[2]);
                    i1=tmp1 / m_shape[2];
                    i2=tmp1 % m_shape[2];
                    m_data_float128_3.find(cur_id)->second(i0,i1,i2) = boost::lexical_cast<long double>(*it);
                    std::cout << m_data_float128_3.find(cur_id)->second(i0,i1,i2) << " ";
                  }
                  break;
                case 48: m_data_float128_4.insert(
                          std::pair<size_t,blitz::Array<long double,4> >
                          ( cur_id, blitz::Array<long double,4>(m_shape[0], m_shape[1], m_shape[2], m_shape[3]) ) 
                          );
                  for( boost::tokenizer<boost::char_separator<char> >::iterator
                    it=tok.begin(); it!=tok.end(); ++it, ++count ) 
                  {
                    i0=count / (m_shape[1]*m_shape[2]*m_shape[3]);
                    tmp1=count - i0*(m_shape[1]*m_shape[2]*m_shape[3]);
                    i1=tmp1 / (m_shape[2]*m_shape[3]);
                    tmp2=tmp1 - i1*(m_shape[2]*m_shape[3]);
                    i2=tmp2 / m_shape[3];
                    i3=tmp2 % m_shape[3];
                    m_data_float128_4.find(cur_id)->second(i0,i1,i2,i3) = boost::lexical_cast<long double>(*it);
                    std::cout << m_data_float128_4.find(cur_id)->second(i0,i1,i2,i3) << " ";
                  }
                  break;
                case 49: m_data_complex64_1.insert(
                          std::pair<size_t,blitz::Array<std::complex<float> ,1> >
                          ( cur_id, blitz::Array<std::complex<float> ,1>(m_shape[0]) ) 
                          );
                  for( boost::tokenizer<boost::char_separator<char> >::iterator
                    it=tok.begin(); it!=tok.end(); ++it, ++count ) 
                  {
                    m_data_complex64_1.find(cur_id)->second(count) = boost::lexical_cast<std::complex<float> >(*it);
                    std::cout << m_data_complex64_1.find(cur_id)->second(count) << " ";
                  }
                  break;
                case 50: m_data_complex64_2.insert(
                          std::pair<size_t,blitz::Array<std::complex<float> ,2> >
                          ( cur_id, blitz::Array<std::complex<float> ,2>(m_shape[0], m_shape[1]) ) 
                          );
                  for( boost::tokenizer<boost::char_separator<char> >::iterator
                    it=tok.begin(); it!=tok.end(); ++it, ++count ) 
                  {
                    i0=count / m_shape[1];
                    i1=count % m_shape[1];
                    m_data_complex64_2.find(cur_id)->second(i0,i1) = boost::lexical_cast<std::complex<float> >(*it);
                    std::cout << m_data_complex64_2.find(cur_id)->second(i0,i1) << " ";
                  }
                  break;
                case 51: m_data_complex64_3.insert(
                          std::pair<size_t,blitz::Array<std::complex<float> ,3> >
                          ( cur_id, blitz::Array<std::complex<float> ,3>(m_shape[0], m_shape[1], m_shape[2]) ) 
                          );
                  for( boost::tokenizer<boost::char_separator<char> >::iterator
                    it=tok.begin(); it!=tok.end(); ++it, ++count ) 
                  {
                    i0=count / (m_shape[1]*m_shape[2]);
                    tmp1=count - i0*(m_shape[1]*m_shape[2]);
                    i1=tmp1 / m_shape[2];
                    i2=tmp1 % m_shape[2];
                    m_data_complex64_3.find(cur_id)->second(i0,i1,i2) = boost::lexical_cast<std::complex<float> >(*it);
                    std::cout << m_data_complex64_3.find(cur_id)->second(i0,i1,i2) << " ";
                  }
                  break;
                case 52: m_data_complex64_4.insert(
                          std::pair<size_t,blitz::Array<std::complex<float> ,4> >
                          ( cur_id, blitz::Array<std::complex<float> ,4>(m_shape[0], m_shape[1], m_shape[2], m_shape[3]) ) 
                          );
                  for( boost::tokenizer<boost::char_separator<char> >::iterator
                    it=tok.begin(); it!=tok.end(); ++it, ++count ) 
                  {
                    i0=count / (m_shape[1]*m_shape[2]*m_shape[3]);
                    tmp1=count - i0*(m_shape[1]*m_shape[2]*m_shape[3]);
                    i1=tmp1 / (m_shape[2]*m_shape[3]);
                    tmp2=tmp1 - i1*(m_shape[2]*m_shape[3]);
                    i2=tmp2 / m_shape[3];
                    i3=tmp2 % m_shape[3];
                    m_data_complex64_4.find(cur_id)->second(i0,i1,i2,i3) = boost::lexical_cast<std::complex<float> >(*it);
                    std::cout << m_data_complex64_4.find(cur_id)->second(i0,i1,i2,i3) << " ";
                  }
                  break;
                case 53: m_data_complex128_1.insert(
                          std::pair<size_t,blitz::Array<std::complex<double> ,1> >
                          ( cur_id, blitz::Array<std::complex<double> ,1>(m_shape[0]) ) 
                          );
                  for( boost::tokenizer<boost::char_separator<char> >::iterator
                    it=tok.begin(); it!=tok.end(); ++it, ++count ) 
                  {
                    m_data_complex128_1.find(cur_id)->second(count) = boost::lexical_cast<std::complex<double> >(*it);
                    std::cout << m_data_complex128_1.find(cur_id)->second(count) << " ";
                  }
                  break;
                case 54: m_data_complex128_2.insert(
                          std::pair<size_t,blitz::Array<std::complex<double> ,2> >
                          ( cur_id, blitz::Array<std::complex<double> ,2>(m_shape[0], m_shape[1]) ) 
                          );
                  for( boost::tokenizer<boost::char_separator<char> >::iterator
                    it=tok.begin(); it!=tok.end(); ++it, ++count ) 
                  {
                    i0=count / m_shape[1];
                    i1=count % m_shape[1];
                    m_data_complex128_2.find(cur_id)->second(i0,i1) = boost::lexical_cast<std::complex<double> >(*it);
                    std::cout << m_data_complex128_2.find(cur_id)->second(i0,i1) << " ";
                  }
                  break;
                case 55: m_data_complex128_3.insert(
                          std::pair<size_t,blitz::Array<std::complex<double> ,3> >
                          ( cur_id, blitz::Array<std::complex<double> ,3>(m_shape[0], m_shape[1], m_shape[2]) ) 
                          );
                  for( boost::tokenizer<boost::char_separator<char> >::iterator
                    it=tok.begin(); it!=tok.end(); ++it, ++count ) 
                  {
                    i0=count / (m_shape[1]*m_shape[2]);
                    tmp1=count - i0*(m_shape[1]*m_shape[2]);
                    i1=tmp1 / m_shape[2];
                    i2=tmp1 % m_shape[2];
                    m_data_complex128_3.find(cur_id)->second(i0,i1,i2) = boost::lexical_cast<std::complex<double> >(*it);
                    std::cout << m_data_complex128_3.find(cur_id)->second(i0,i1,i2) << " ";
                  }
                  break;
                case 56: m_data_complex128_4.insert(
                          std::pair<size_t,blitz::Array<std::complex<double> ,4> >
                          ( cur_id, blitz::Array<std::complex<double> ,4>(m_shape[0], m_shape[1], m_shape[2], m_shape[3]) ) 
                          );
                  for( boost::tokenizer<boost::char_separator<char> >::iterator
                    it=tok.begin(); it!=tok.end(); ++it, ++count ) 
                  {
                    i0=count / (m_shape[1]*m_shape[2]*m_shape[3]);
                    tmp1=count - i0*(m_shape[1]*m_shape[2]*m_shape[3]);
                    i1=tmp1 / (m_shape[2]*m_shape[3]);
                    tmp2=tmp1 - i1*(m_shape[2]*m_shape[3]);
                    i2=tmp2 / m_shape[3];
                    i3=tmp2 % m_shape[3];
                    m_data_complex128_4.find(cur_id)->second(i0,i1,i2,i3) = boost::lexical_cast<std::complex<double> >(*it);
                    std::cout << m_data_complex128_4.find(cur_id)->second(i0,i1,i2,i3) << " ";
                  }
                  break;
                case 57: m_data_complex256_1.insert(
                          std::pair<size_t,blitz::Array<std::complex<long double> ,1> >
                          ( cur_id, blitz::Array<std::complex<long double> ,1>(m_shape[0]) ) 
                          );
                  for( boost::tokenizer<boost::char_separator<char> >::iterator
                    it=tok.begin(); it!=tok.end(); ++it, ++count ) 
                  {
                    m_data_complex256_1.find(cur_id)->second(count) = boost::lexical_cast<std::complex<long double> >(*it);
                    std::cout << m_data_complex256_1.find(cur_id)->second(count) << " ";
                  }
                  break;
                case 58: m_data_complex256_2.insert(
                          std::pair<size_t,blitz::Array<std::complex<long double> ,2> >
                          ( cur_id, blitz::Array<std::complex<long double> ,2>(m_shape[0], m_shape[1]) ) 
                          );
                  for( boost::tokenizer<boost::char_separator<char> >::iterator
                    it=tok.begin(); it!=tok.end(); ++it, ++count ) 
                  {
                    i0=count / m_shape[1];
                    i1=count % m_shape[1];
                    m_data_complex256_2.find(cur_id)->second(i0,i1) = boost::lexical_cast<std::complex<long double> >(*it);
                    std::cout << m_data_complex256_2.find(cur_id)->second(i0,i1) << " ";
                  }
                  break;
                case 59: m_data_complex256_3.insert(
                          std::pair<size_t,blitz::Array<std::complex<long double> ,3> >
                          ( cur_id, blitz::Array<std::complex<long double> ,3>(m_shape[0], m_shape[1], m_shape[2]) ) 
                          );
                  for( boost::tokenizer<boost::char_separator<char> >::iterator
                    it=tok.begin(); it!=tok.end(); ++it, ++count ) 
                  {
                    i0=count / (m_shape[1]*m_shape[2]);
                    tmp1=count - i0*(m_shape[1]*m_shape[2]);
                    i1=tmp1 / m_shape[2];
                    i2=tmp1 % m_shape[2];
                    m_data_complex256_3.find(cur_id)->second(i0,i1,i2) = boost::lexical_cast<std::complex<long double> >(*it);
                    std::cout << m_data_complex256_3.find(cur_id)->second(i0,i1,i2) << " ";
                  }
                  break;
                case 60: m_data_complex256_4.insert(
                          std::pair<size_t,blitz::Array<std::complex<long double> ,4> >
                          ( cur_id, blitz::Array<std::complex<long double> ,4>(m_shape[0], m_shape[1], m_shape[2], m_shape[3]) ) 
                          );
                  for( boost::tokenizer<boost::char_separator<char> >::iterator
                    it=tok.begin(); it!=tok.end(); ++it, ++count ) 
                  {
                    i0=count / (m_shape[1]*m_shape[2]*m_shape[3]);
                    tmp1=count - i0*(m_shape[1]*m_shape[2]*m_shape[3]);
                    i1=tmp1 / (m_shape[2]*m_shape[3]);
                    tmp2=tmp1 - i1*(m_shape[2]*m_shape[3]);
                    i2=tmp2 / m_shape[3];
                    i3=tmp2 % m_shape[3];
                    m_data_complex256_4.find(cur_id)->second(i0,i1,i2,i3) = boost::lexical_cast<std::complex<long double> >(*it);
                    std::cout << m_data_complex256_4.find(cur_id)->second(i0,i1,i2,i3) << " ";
                  }
                  break;
                default:
                  break;
              }

              if(count < total) {
                error << "Only " <<  count << " elements have been found " <<
                  "instead of " << total << " expected." << std::endl;
                throw Exception();        
              }

              std::cout << std::endl;

              //ar = new ArrayXML(cur);
              //m_arrayset.insert ( std::pair<size_t,const ArraysetXML*>(ar->getId(), ar) );
            }
            else {
              //m_data_filenames.insert( std::pair<size_t,s_file_loader(s_filename, s_loader)>);
            }


          }
          cur = cur->next;
        }
      }
      // load the arrayset from a file
      else {
      }  
      
      std::cout << std::endl;
    }


    void ArraysetXML::setArraydata(const xmlNodePtr& array) {
    }




    DatasetXML::DatasetXML(char *filename) {
      parseFile(filename);
    }


    void DatasetXML::parseFile(char *filename) {
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

      // Parse Arraysets
      ArraysetXML* ar;
      cur = cur->xmlChildrenNode;
      while (cur != 0) {
        // Process an arrayset
        if ((!xmlStrcmp(cur->name, xmlCharStrdup(db::arrayset)))) {
          ar = new ArraysetXML(cur);
          m_arrayset.insert( std::pair<size_t,ArraysetXML*>(ar->getId(), ar) );
        }
        cur = cur->next;
      }
    }


    DatasetXML::~DatasetXML() {
      // Free libxml2 allocated memory
      if(m_doc) xmlFreeDoc(m_doc);
      
      // Remove the arrayset
      std::map<size_t, ArraysetXML*>::iterator it;
      for ( it=m_arrayset.begin() ; it != m_arrayset.end(); ++it ) {
        ArraysetXML *ar=it->second;
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

