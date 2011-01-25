/**
 * @file src/cxx/core/src/XMLWriter.cc
 * @author <a href="mailto:Laurent.El-Shafey@idiap.ch">Laurent El Shafey</a>
 *
 * @brief Implements the XML writer for a dataset.
 */

#include "core/XMLWriter.h"
#include "core/Exception.h"

namespace Torch {
  namespace core {

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

      // Write name attribute if any
      if(dataset.getName().compare("") )
        xmlNewProp( rootnode, (const xmlChar*)db::name, 
          (const xmlChar*)dataset.getName().c_str() );

      // Write version attribute if any
      if(dataset.getVersion() != 0)
        xmlNewProp( rootnode, (const xmlChar*)db::version, (const xmlChar*)
          (boost::lexical_cast<std::string>(dataset.getVersion())).c_str() );

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
      bool content_inline, int precision, bool scientific) 
    {
      // Create the Arrayset node
      xmlNodePtr arraysetnode; 
      if( a.getFilename().compare("") && !content_inline)
        arraysetnode = 
          xmlNewDocNode(doc, 0, (const xmlChar*)db::external_arrayset, 0);
      else
        arraysetnode = xmlNewDocNode(doc, 0, (const xmlChar*)db::arrayset, 0);

      // Write id attribute
      xmlNewProp( arraysetnode, (const xmlChar*)db::id, (const xmlChar*)
        (boost::lexical_cast<std::string>(a.getId())).c_str() );

      // Write elementtype attribute
      std::string str;
      switch(a.getElementType()) {
        case array::t_bool:
          str = db::t_bool; break;
        case array::t_int8:
          str = db::t_int8; break;
        case array::t_int16:
          str = db::t_int16; break;
        case array::t_int32:
          str = db::t_int32; break;
        case array::t_int64:
          str = db::t_int64; break;
        case array::t_uint8:
          str = db::t_uint8; break;
        case array::t_uint16:
          str = db::t_uint16; break;
        case array::t_uint32:
          str = db::t_uint32; break;
        case array::t_uint64:
          str = db::t_uint64; break;
        case array::t_float32:
          str = db::t_float32; break;
        case array::t_float64:
          str = db::t_float64; break;
        case array::t_float128:
          str = db::t_float128; break;
        case array::t_complex64:
          str = db::t_complex64; break;
        case array::t_complex128:
          str = db::t_complex128; break;
        case array::t_complex256:
          str = db::t_complex256; break;
        default:
          throw Exception();
          break;
      }    
      xmlNewProp( arraysetnode, (const xmlChar*)db::elementtype, (const xmlChar*)
        str.c_str() );

      // Write shape attribute
      const size_t* shape = a.getShape();
      str = boost::lexical_cast<std::string>(shape[0]);
      for(size_t i=1; i<a.getNDim(); ++i)
        str += " " + boost::lexical_cast<std::string>(shape[i]);
      xmlNewProp( arraysetnode, (const xmlChar*)db::shape, (const xmlChar*)
        str.c_str() );

      // Write role attribute
      xmlNewProp( arraysetnode, (const xmlChar*)db::role, (const xmlChar*)
        a.getRole().c_str() );

      // Write file and loader attributes if any
      if( a.getFilename().compare("") && !content_inline)
      {
        // Write file attribute
        xmlNewProp( arraysetnode, (const xmlChar*)db::file, (const xmlChar*)
          a.getFilename().c_str() );

        // Write loader attribute
        str = "";
        switch( a.getLoader() )
        {
          case l_blitz:
            str = db::l_blitz; break;
          case l_tensor:
            str = db::l_tensor; break;
          case l_bindata:
            str = db::l_bindata; break;
          default:
            throw Exception();
            break;
        }
        xmlNewProp( arraysetnode, (const xmlChar*)db::loader, (const xmlChar*)
          str.c_str() );
      }

      // Create Array nodes
      for(Arrayset::const_iterator it=a.begin(); it!=a.end(); 
        ++it)
      {
        xmlAddChild( arraysetnode, writeArray( doc, *it->second, 
          content_inline, precision, scientific) );
      }

      return arraysetnode;
    }


    xmlNodePtr XMLWriter::writeArray( xmlDocPtr doc, const Array& a, 
      bool content_inline, int precision, bool scientific) 
    {
      // Create the Arrayset node
      xmlNodePtr arraynode; 
      if( a.getFilename().compare("") && !content_inline)
        arraynode = 
          xmlNewDocNode(doc, 0, (const xmlChar*)db::external_array, 0);
      else {
        // Prepare the string stream
        std::stringstream content;
        content << std::setprecision(precision);
        if( scientific)
          content << std::scientific;

        // Cast the data and call the writing function
        switch(a.getParentArrayset().getElementType()) {
          case array::t_bool:
            writeData( content, 
              reinterpret_cast<const bool*>(a.getStorage()), 
              a.getParentArrayset().getNElem()); break;
          case array::t_int8:
            writeData( content, 
              reinterpret_cast<const int8_t*>(a.getStorage()), 
              a.getParentArrayset().getNElem()); break;
          case array::t_int16:
            writeData( content, 
              reinterpret_cast<const int16_t*>(a.getStorage()),
              a.getParentArrayset().getNElem()); break;
          case array::t_int32:
            writeData( content, 
              reinterpret_cast<const int32_t*>(a.getStorage()), 
              a.getParentArrayset().getNElem()); break;
          case array::t_int64:
            writeData( content, 
              reinterpret_cast<const int64_t*>(a.getStorage()), 
              a.getParentArrayset().getNElem()); break;
          case array::t_uint8:
            writeData( content, 
              reinterpret_cast<const uint8_t*>(a.getStorage()), 
              a.getParentArrayset().getNElem()); break;
          case array::t_uint16:
            writeData( content, 
              reinterpret_cast<const uint16_t*>(a.getStorage()), 
              a.getParentArrayset().getNElem()); break;
          case array::t_uint32:
            writeData( content, 
              reinterpret_cast<const uint32_t*>(a.getStorage()), 
              a.getParentArrayset().getNElem()); break;
          case array::t_uint64:
            writeData( content, 
              reinterpret_cast<const uint64_t*>(a.getStorage()), 
              a.getParentArrayset().getNElem()); break;
          case array::t_float32:
            writeData( content, 
              reinterpret_cast<const float*>(a.getStorage()), 
              a.getParentArrayset().getNElem()); break;
          case array::t_float64:
            writeData( content, 
              reinterpret_cast<const double*>(a.getStorage()), 
              a.getParentArrayset().getNElem()); break;
          case array::t_float128:
            writeData( content, 
              reinterpret_cast<const long double*>(a.getStorage()), 
              a.getParentArrayset().getNElem()); break;
          case array::t_complex64:
            writeData( content, 
              reinterpret_cast<const std::complex<float>*>(a.getStorage()), 
              a.getParentArrayset().getNElem()); break;
          case array::t_complex128:
            writeData( content, 
              reinterpret_cast<const std::complex<double>*>(a.getStorage()),
              a.getParentArrayset().getNElem()); break;
          case array::t_complex256:
            writeData( content, 
              reinterpret_cast<const std::complex<long double>*>(
              a.getStorage()), a.getParentArrayset().getNElem()); break;
          default:
            throw Exception();
            break;
        }
 
        TDEBUG3("Inline content: " << content.str());
        arraynode = xmlNewDocNode(doc, 0, (const xmlChar*)db::array, 
          (const xmlChar*)(content.str().c_str()));
      }

      // Write id attribute
      xmlNewProp( arraynode, (const xmlChar*)db::id, (const xmlChar*)
        (boost::lexical_cast<std::string>(a.getId())).c_str() );

      if( a.getFilename().compare("") && !content_inline)
      {
        // Write file attribute
        xmlNewProp( arraynode, (const xmlChar*)db::file, (const xmlChar*)
          a.getFilename().c_str() );

        // Write loader attribute
        std::string str;
        switch( a.getLoader() )
        {
          case l_blitz:
            str = db::l_blitz; break;
          case l_tensor:
            str = db::l_tensor; break;
          case l_bindata:
            str = db::l_bindata; break;
          default:
            throw Exception();
            break;
        }
        xmlNewProp( arraynode, (const xmlChar*)db::loader, (const xmlChar*)
          str.c_str() );
      }

      return arraynode;
    }


    xmlNodePtr XMLWriter::writeRelationset( xmlDocPtr doc, const Relationset& r,
      bool content_inline) 
    {
      // Create the Relationset node
      xmlNodePtr relationsetnode = 
        xmlNewDocNode(doc, 0, (const xmlChar*)db::relationset, 0);

      // Write name attribute
      xmlNewProp( relationsetnode, (const xmlChar*)db::name, 
          (const xmlChar*)r.getName().c_str() );

      // Add the Rule nodes to the relationset node
      for(Relationset::rule_const_iterator it=r.rule_begin(); 
        it!=r.rule_end(); ++it)
      {
        xmlAddChild( relationsetnode, writeRule( doc, *it->second) );
      }

      // Add the Relation nodes to the relationset node
      for(Relationset::const_iterator it=r.begin(); it!=r.end(); ++it)
      {
        xmlAddChild( relationsetnode, writeRelation( doc, *it->second) );
      }

      return relationsetnode;
    }
    

    xmlNodePtr XMLWriter::writeRule( xmlDocPtr doc, const Rule& r) {
      // Create the Rule node
      xmlNodePtr rulenode = xmlNewDocNode(doc,0, (const xmlChar*)db::rule, 0);

      // Write arrayset-role attribute
      xmlNewProp( rulenode, (const xmlChar*)db::arrayset_role, 
          (const xmlChar*)r.getArraysetRole().c_str() );
      // Write min attribute
      xmlNewProp( rulenode, (const xmlChar*)db::min, (const xmlChar*)
        (boost::lexical_cast<std::string>(r.getMin())).c_str() );
      // Write max attribute
      xmlNewProp( rulenode, (const xmlChar*)db::max, (const xmlChar*)
        (boost::lexical_cast<std::string>(r.getMax())).c_str() );

      return rulenode;
    }
    
    xmlNodePtr XMLWriter::writeRelation( xmlDocPtr doc, const Relation& r) {
      // Create the Relation node
      xmlNodePtr relationnode = 
        xmlNewDocNode(doc, 0, (const xmlChar*)db::relation, 0);

      // Write id attribute
      xmlNewProp( relationnode, (const xmlChar*)db::id, (const xmlChar*)
        (boost::lexical_cast<std::string>(r.getId())).c_str() );

      // Add the Member nodes to the relation node
      for(Relation::const_iterator it=r.begin(); it!=r.end(); ++it)
      {
        xmlAddChild( relationnode, writeMember( doc, *it->second) );
      }
      return relationnode;
    }

    xmlNodePtr XMLWriter::writeMember( xmlDocPtr doc, const Member& m) {
      // Create the Member node
      xmlNodePtr membernode; 
      if( m.getArrayId() != 0)
        membernode = xmlNewDocNode(doc, 0, (const xmlChar*)db::member, 0);
      else
        membernode = 
          xmlNewDocNode(doc, 0, (const xmlChar*)db::arrayset_member, 0);

      // Write arrayset-id attribute
      xmlNewProp( membernode, (const xmlChar*)db::arrayset_id, (const xmlChar*)
        (boost::lexical_cast<std::string>(m.getArraysetId())).c_str() );

      // Write array-id attribute if any
      if( m.getArrayId() != 0)
        xmlNewProp( membernode, (const xmlChar*)db::array_id, (const xmlChar*)
          (boost::lexical_cast<std::string>(m.getArrayId())).c_str() );

      return membernode;
    }

  }
}

